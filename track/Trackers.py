"""
Written by Matteo Dunnhofer - 2019

Class that defines the A3C tracker
"""
import sys
sys.path.insert(0, '..')
import copy
import numpy as np
import torch
from model.StudentModel import StudentModel
from data.DataTransformer import DataTransformer
import utils as ut
#from trackers.SiamFC.siamfc import TrackerSiamFC
from trackers.ResultsTracker import ResultsTracker

from got10k.trackers import Tracker

class TRAS(object):

    def __init__(self, cfg):
        super(TRAS, self).__init__()

        self.cfg = cfg
        self.data_transformer = DataTransformer(self.cfg)

        self.device = torch.device('cuda', 0) if self.cfg.USE_GPU else torch.device('cpu')

        self.student = StudentModel(self.cfg).to(self.device)
        self.lstm_update = self.cfg.LSTM_UPDATE

        ckpt = torch.load(self.cfg.CKPT_PATH, map_location=self.device)
        if type(ckpt) is dict:
            ckpt = ckpt['model_state_dict']
        self.student.load_state_dict(ckpt)

        self.student.eval()



    def init(self, image, box, **kwargs):
        self.curr_bb = box
        self.prev_img = image

        self.student_state = self.student.init_state(self.device)

        self.step = 1


    def update(self, image):

        # LSTM STATE UPDATE
        if self.cfg.LSTM_UPDATE and (self.step % self.cfg.SEQ_LENGTH == 0):
            self.student_state = self.first_student_state

        bb = ut.get_crop_bb(copy.deepcopy(self.curr_bb), image.size[0], image.size[1], self.cfg.CONTEXT_FACTOR)

        state1 = self.data_transformer.preprocess_img(self.prev_img, bb).to(self.device).unsqueeze(0)
        state2 = self.data_transformer.preprocess_img(image, bb).to(self.device).unsqueeze(0)

        feats, self.student_state = self.student.get_feats(state1, state2, self.student_state, self.device)
        action = self.student.actor_policy(feats)
        action = torch.clamp(action, -1.0, 1.0)

        bbox = ut.denorm_action(action.data.cpu().numpy()[0], self.curr_bb)

        if self.step == 1:
            self.first_student_state = (self.student_state[0].clone(), self.student_state[1].clone())

        self.step += 1

        self.curr_bb = ut.clip_bb(bbox, image)
        self.prev_img = image

        return np.array(self.curr_bb)



class TRAST(object):

    def __init__(self, cfg):
        super(TRAST, self).__init__()

        self.cfg = cfg
        self.data_transformer = DataTransformer(self.cfg)

        self.device = torch.device('cuda', 0) if self.cfg.USE_GPU else torch.device('cpu')

        self.student = StudentModel(self.cfg).to(self.device)
        self.lstm_update = self.cfg.LSTM_UPDATE

        ckpt = torch.load(self.cfg.CKPT_PATH, map_location=self.device)
        if type(ckpt) is dict:
            ckpt = ckpt['model_state_dict']
        self.student.load_state_dict(ckpt)

        self.student.eval()

        if self.cfg.USE_RESULTS:
            self.teacher = ResultsTracker(self.cfg.TRAST_TEACHER)
        else:
            # instantiate here your teacher implementation
            self.teacher = YourTracker()



    def init(self, image, box, **kwargs):
        # initialize teacher trackers
        self.teacher_failed_on_init = False
        try:
            self.teacher.init(image, box, **kwargs)
        except Exception as e:
            print(e)
            print('Teacher {} failure on initialization'.format(self.teacher.name))
            self.teacher_failed_on_init = True

        self.curr_bb = box
        self.prev_img = image

        self.student_state = copy.deepcopy(self.student.init_state(self.device))
        self.teacher_state = copy.deepcopy(self.student.init_state(self.device))

        self.teacher_curr_bb = copy.deepcopy(box)

        self.step = 1



    def update(self, image):

        teacher_pred = np.zeros(4, dtype=np.float32)
        teacher_failed = False

        try:
            teacher_pred = np.array(self.teacher.update(image))
        except Exception as e:
            print('Teacher {} failure'.format(self.teacher.name))
            teacher_failed = True

        # LSTM's state update
        if self.cfg.LSTM_UPDATE and (self.step % self.cfg.SEQ_LENGTH == 0):
            self.student_state = self.first_student_state

            if (not self.teacher_failed_on_init):
                self.teacher_state = (self.first_teacher_state[0].clone(), self.first_teacher_state[1].clone())

        # tracker prediction
        bb = ut.get_crop_bb(copy.deepcopy(self.curr_bb), image.size[0], image.size[1], self.cfg.CONTEXT_FACTOR)

        state1 = self.data_transformer.preprocess_img(self.prev_img, bb).to(self.device).unsqueeze(0)
        state2 = self.data_transformer.preprocess_img(image, bb).to(self.device).unsqueeze(0)

        # get a3c tracker bounding box estimate
        feats, self.student_state = self.student.get_feats(state1, state2, self.student_state, self.device)
        action = self.student.actor_policy(feats)
        action = torch.clamp(action, -1.0, 1.0)
        bbox = ut.denorm_action(action.data.cpu().numpy()[0], self.curr_bb)

        student_value = self.student.critic(feats).data.cpu().numpy()[0]

        # evaluate teacher
        teacher_value = -np.inf
        if not self.teacher_failed_on_init:
            bb = ut.get_crop_bb(self.teacher_curr_bb, image.size[0], image.size[1], self.cfg.CONTEXT_FACTOR)

            #prev_img = copy.deepcopy(self.prev_img)
            #curr_img = copy.deepcopy(image)

            state1 = self.data_transformer.preprocess_img(self.prev_img, bb).to(self.device).unsqueeze(0)
            state2 = self.data_transformer.preprocess_img(image, bb).to(self.device).unsqueeze(0)

            feats, self.teacher_state = self.student.get_feats(state1, state2, self.teacher_state, self.device)
            teacher_value = self.student.critic(feats).data.cpu().numpy()[0]

            #_, expert_value, n_expert_model_state = self.model(state1, state2, self.expert_model_states[i], self.device)
            #self.expert_model_states[i] = (n_expert_model_state[0].clone(), n_expert_model_state[1].clone())
            #expert_values[i] = expert_value.data.cpu().numpy()[0, 0]

            self.teacher_curr_bb = copy.deepcopy(teacher_pred)


        self.prev_img = copy.deepcopy(image)

        # select student's or teacher's bounding-box prediction
        if (not self.teacher_failed_on_init) and (not teacher_failed):
            if student_value >= teacher_value:
                self.curr_bb = copy.deepcopy(bbox)
            else:
                self.curr_bb = copy.deepcopy(teacher_pred)
        else:
            self.curr_bb = copy.deepcopy(bbox)

        self.curr_bb = ut.clip_bb(self.curr_bb, image)

        if self.step == 1:
            self.first_student_state = (self.student_state[0].clone(), self.student_state[1].clone())
            self.first_teacher_state = (self.teacher_state[0].clone(), self.teacher_state[1].clone())

        self.step += 1

        return self.curr_bb




class TRASFUST(object):

    def __init__(self, cfg):
        super(TRASFUST, self).__init__()

        self.cfg = cfg
        self.data_transformer = DataTransformer(self.cfg)

        self.device = torch.device('cuda', 0) if self.cfg.USE_GPU else torch.device('cpu')

        self.student = StudentModel(self.cfg).to(self.device)
        self.lstm_update = self.cfg.LSTM_UPDATE

        ckpt = torch.load(self.cfg.CKPT_PATH, map_location=self.device)
        if type(ckpt) is dict:
            ckpt = ckpt['model_state_dict']
        self.student.load_state_dict(ckpt)

        self.student.eval()

        self.teachers = []
        for teacher_id in self.cfg.TRASFUST_TEACHERS:

            if self.cfg.USE_RESULTS:
                teacher = ResultsTracker(teacher_id)
            else:
                teacher = YourTracker()

            self.teachers.append(teacher)

        self.n_teachers = len(self.teachers)

    def init(self, image, box, **kwargs):

        # initialize teacher trackers
        self.teachers_failed_on_init = [False] * self.n_teachers
        for i, tt in enumerate(self.teachers):
            try:
                tt.init(image, box, **kwargs)
            except Exception as e:
                print(e)
                print('Teacher {} failure on initialization'.format(tt.name))
                self.teachers_failed_on_init[i] = True

        self.curr_bb = copy.deepcopy(box)
        self.prev_img = copy.deepcopy(image)


        self.teacher_states = []
        self.teacher_curr_bbs = []
        for _ in range(self.n_teachers):
            self.teacher_states.append(copy.deepcopy(self.student.init_state(self.device)))
            self.teacher_curr_bbs.append(copy.deepcopy(box))

        self.step = 1


    def update(self, image):

        teacher_preds = np.zeros((self.n_teachers, 4), dtype=np.float32)
        teachers_failed = [False] * self.n_teachers

        for i, tt in enumerate(self.teachers):
            try:
                teacher_pred = tt.update(image)
                teacher_preds[i] = np.array(teacher_pred)
            except Exception as e:
                print('Teacher {} failure'.format(tt.name))
                teachers_failed[i] = True
                teacher_preds[i] = np.zeros(4)

        # LSTM's state update
        if self.cfg.LSTM_UPDATE and (self.step % self.cfg.SEQ_LENGTH == 0):
            for i in range(self.n_teachers):
                if (not self.teachers_failed_on_init[i]):
                    self.teacher_states[i] = (self.first_teacher_states[i][0].clone(), self.first_teacher_states[i][1].clone())

        # evaluate each tracker prediction
        teacher_values = np.zeros(self.n_teachers) - np.inf
        for i in range(self.n_teachers):
            if (not self.teachers_failed_on_init[i]):
                bb = ut.get_crop_bb(copy.deepcopy(self.teacher_curr_bbs[i]), image.size[0], image.size[1], self.cfg.CONTEXT_FACTOR)

                #prev_img = copy.deepcopy(self.prev_img)
                #curr_img = copy.deepcopy(image)

                state1 = self.data_transformer.preprocess_img(self.prev_img, bb).to(self.device).unsqueeze(0)
                state2 = self.data_transformer.preprocess_img(image, bb).to(self.device).unsqueeze(0)

                feats, n_teacher_state = self.student.get_feats(state1, state2, self.teacher_states[i], self.device)
                teacher_value = self.student.critic(feats).data.cpu().numpy()[0]
                teacher_values[i] = teacher_value

                self.teacher_states[i] = (n_teacher_state[0].clone(), n_teacher_state[1].clone())

                self.teacher_curr_bbs[i] = copy.deepcopy(teacher_preds[i])

        self.prev_img = copy.deepcopy(image)

        best_teacher_idx = np.argmax(teacher_values)
        self.curr_bb = copy.deepcopy(teacher_preds[best_teacher_idx])

        self.curr_bb = ut.clip_bb(self.curr_bb, image)

        if self.step == 1:
            self.first_teacher_states = []
            for i in range(self.n_teachers):
                self.first_teacher_states.append((self.teacher_states[i][0].clone(), self.teacher_states[i][1].clone()))

        self.step += 1

        return self.curr_bb


############################################################################
######## Tracker definitions to use with the GOT-10k toolkit #############

class Tracker_got10k(Tracker):

    def __init__(self, tracker_name, cfg):

        super(Tracker_got10k, self).__init__(name=tracker_name, is_deterministic=True)

        if tracker_name == 'TRAS' or tracker_name == 'A3CT':
            self.tracker = TRAS(cfg)
        elif tracker_name == 'TRAST' or tracker_name == 'A3CTD':
            self.tracker = TRAST(cfg)
        if tracker_name == 'TRASFUST':
            self.tracker = TRASFUST(cfg)

    def init(self, image, box, **kwargs):
        self.tracker.init(image, box, **kwargs)

    def update(self, image):
        return self.tracker.update(image)