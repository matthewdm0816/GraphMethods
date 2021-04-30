import os, sys, random, math, json
import torch

class MilestoneRecorder():
    def __init__(self, comment: str):
        self.path = comment


    def add_model(self, obj: dict, id, path=None):
        """
        Saved model must has state_dict() method
        """
        id = id if isinstance(id, str) else str(id)
        f = self.path if path is None else path
        for name, model in obj:
            assert isinstance(name, str), 'Saved object type description must be string!'
            f_save = os.path.join(f, name, '%s.save' % (id))
            torch.save(model.state_dict(), f_save)


    def load_model(self, obj: dict, id, path=None):
        id = id if isinstance(id, str) else str(id)
        f = self.path if path is None else path
        for name, model in obj:
            assert isinstance(name, str), 'Saved object type description must be string!'
            f_save = os.path.join(f, name, '%s.save' % (id))
            model.load_state_dict(f_save)


    def add_unique_dict(self, obj: dict, name: str, path=None):
        f_path = self.path if path is None else path
        f_save = os.path.join(f_path, '%s.save' % (name))
        with open(f_save, 'w') as f:
            json.dump(obj, f)


    def load_unique_dict(self, obj: dict, name: str, path=None):
        f_path = self.path if path is None else path
        f_save = os.path.join(f_path, '%s.save' % (name))
        with open(f_save, 'w') as f:
            return json.load(f)


    def add_tensor(self, obj: dict, path=None):
        f = self.path if path is None else path
        for name, t in obj:
            assert isinstance(name, str), 'Saved object type description must be string!'
            assert isinstance(t, torch.Tensor), 'Saved object must be tensor!'
            f_save = os.path.join(f, '%s.save' % (name))
            torch.save(t, f_save)


    def load_tensor(self, obj: list, path=None):
        f = self.path if path is None else path
        tensors = dict()
        for name in obj:
            assert isinstance(name, str), 'Saved object type description must be string!'
            f_save = os.path.join(f, '%s.save' % (name))
            tensors[name] = torch.load(f_save)
        return tensors

