from bactorcanet.config import load_config
from bactorcanet.training import Trainer

config = load_config("configs/default.yaml")
trainer = Trainer(config)
trainer.fit()
trainer.test()
