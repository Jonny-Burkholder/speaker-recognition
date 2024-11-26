import torch
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Model
from pyannote.audio import Inference

local_model = Model.from_pretrained("./model/pytorch_model.bin")

# model = PretrainedSpeakerEmbedding(
#     local_model,
#     device=torch.device("cuda"))

# create an inferenece instance I guess?
inference = Inference(local_model, window="whole")

# extract embedding for a speaker from file
embedding1 = inference("1.wav")
embedding2 = inference("2.wav")

print("Shape of embedding1:", embedding1.shape)
print("Shape of embedding2:", embedding2.shape)

# Ensure embeddings are 2-dimensional
embedding1 = embedding1.reshape(1, -1)
embedding2 = embedding2.reshape(1, -1)

print("embedding a \n")
print(embedding1)
print("\n\n\n\n")
print("embedding b")
print(embedding2)

# compare using cosine similarity
from scipy.spatial.distance import cdist
distance = cdist(embedding1, embedding2, metric="cosine")

print(distance)