from bactorcanet.inference import Predictor

predictor = Predictor.from_checkpoint("runs/default/best.pt")
print(predictor.predict_image("data/test/image_001.png"))
