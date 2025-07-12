import os

class ExperimentState:

    @classmethod
    @property
    def dropout_trans(cls):
        return os.environ.get("dropout_trans", True) not in [0, "0", "f", "F", "false", "False"]
    
    @classmethod
    @property
    def ct_only(cls):
        return os.environ.get("ct_only", False) not in [0, "0", "f", "F", "false", "False"]
    
    @classmethod
    @property
    def mri_only(cls):
        return os.environ.get("mri_only", False) not in [0, "0", "f", "F", "false", "False"]
    
    @classmethod
    @property
    def no_mirror_neither_leftright(cls):
        return os.environ.get("no_mirror_neither_leftright", False) not in [0, "0", "f", "F", "false", "False"]
    
    @classmethod
    @property
    def reg_rigid_only(cls):
        return os.environ.get("reg_rigid_only", True) not in [0, "0", "f", "F", "false", "False"]
    
    @classmethod
    @property
    def mem_optimized(cls):
<<<<<<< HEAD
<<<<<<< HEAD
        return os.environ.get("mem_optimized", False) not in [0, "0", "f", "F", "false", "False"]
=======
        return os.environ.get("mem_optimized", True) not in [0, "0", "f", "F", "false", "False"]
>>>>>>> 3eb5592 (environment variable manager for options of the training)
=======
        return os.environ.get("mem_optimized", False) not in [0, "0", "f", "F", "false", "False"]
>>>>>>> 92b1128 ( turning memopt by default to False since we only should use it for inference)

    @classmethod
    @property
    def epochs(cls):
        return int(os.environ.get("epochs", 1000))
    
    @classmethod
    @property
    def nnunet_std(cls):
        return os.environ.get("nnunet_std", False) not in [0, "0", "f", "F", "false", "False"]


    
if __name__ == "__main__":
    print(ExperimentState.epochs)
    print(ExperimentState.mem_optimized)
    print(ExperimentState.__dict__)
<<<<<<< HEAD
=======

>>>>>>> 3eb5592 (environment variable manager for options of the training)
