import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np

from modeling_clip import CLIP
clip_model = CLIP(device="cpu")

text_prompts = {
"multiple_persons": [
    "A photo of multiple individuals working together.",
    "A photo of several people engaged in conversation.",
    "A photo of a group of people collaborating on a project.",
    "A photo of numerous persons attending a meeting.",
    "A photo of various individuals interacting",
    "A photo of several persons working on a project.",
    # "A photo of a collection of individuals participating",
    "A photo of numerous individuals working",
    "A photo of many people collaborating on a project.",
    # "A photo of a pair of people discussing.",
    # "A photo of numerous persons attending a conference.",
    "A photo of many individuals engaged in an activity.",
    "A photo of numerous people interacting",
    "A photo of many persons involved in a brainstorming session.",
    # "A photo of a pair of individuals collaborating on a task.",
    # "A photo of numerous people contributing to a team effort.", 
    "A photo of many persons in a busy co-working space.", 

    "A photo of a person engaging in a conversation with another individual.",
    "A photo capturing a lively discussion among a group of people.",
    "A photo capturing a discussion among several individuals.",
    "A photo of a person speaking with another person.",

    "A photo of multiple individuals connecting in a meeting.",
    "A photo of multiple people communicating over a task.",
    "A photo of multiple people working together on a project.",
    "A photo of multiple people speaking",
    "A photo of multiple people discussing",
    "A photo of multiple people interacting",
    "A photo of multiple people talking"
    "A photo of multiple people talking in a brainstorming session.",
    "A photo of multiple individuals getting together for a collaboration.",
    # "A photo of multiple people discussing a strategy.",
    "A photo of multiple individuals coordinating on a group effort.",
    # "A photo of multiple people collaborating as a team.",

    "A photo of many people speaking",
    "A photo of many people discussing",
    "A photo of many people interacting",
    "A photo of many people talking"
    "A photo of many individuals connecting in a discussion.",
    "A photo of many people working together on a project.",
    # "A photo of many people speaking during a conference.",
    "A photo of many people communicating in a lively discussion.",
    "A photo of many people communicating during a meeting.",
    "A photo of many people talking over a shared activity.",
    "A photo of many individuals getting together for a workshop.",
    # "A photo of many people discussing ideas in a group setting.",
    "A photo of many people coordinating on a task.",
    # "A photo of many individuals gang up for a team effort.",


    "A photo of several people speaking",
    "A photo of several people discussing",
    "A photo of several people interacting",
    "A photo of several people talking"
    "A photo of several individuals connecting in a discussion.",
    # "A photo of several people communicating during a presentation.",
    "A photo of several people working together on a task.",
    # "A photo of several people speaking at a panel.",
    "A photo of several people talking during a meeting.",
    "A photo of several individuals getting together for a project.",
    "A photo of several people discussing their ideas.",
    "A photo of several individuals coordinating a joint effort.",
    "A photo of several people collaborating effectively.",

    "A photo of numerous people speaking",
    "A photo of numerous people discussing",
    "A photo of numerous people interacting",
    "A photo of numerous people talking"
    # "A photo of numerous individuals connecting at an event.",
    "A photo of numerous people communicating in a group.",
    "A photo of numerous people working together on an initiative.",
    # "A photo of numerous people speaking at a conference.",
    # "A photo of numerous people talking during a workshop.",
    "A photo of numerous individuals getting together for a discussion.",
    "A photo of numerous people discussing a project.",
    "A photo of numerous people coordinating on a task.",

    "A photo of a collective of numerous people speaking.",
    "A photo of a collective of numerous people discussing.",
    "A photo of a collective of numerous people interacting.",
    "A photo of a collective of numerous people talking.",
    "A photo of a collective of individuals working together.",
    "A photo of a collective of people communicating in a group.",
    "A photo of a collective of people collaborating on a project.",
    # "A photo of a collective of people speaking at a forum.",
    "A photo of a collective of people talking during a strategy session.",
    "A photo of a collective of individuals getting together for a meeting.",
    "A photo of a collective of people discussing a plan.",
    "A photo of a collective of individuals coordinating on a task.",
    "A photo of a collective of people collaborating on a goal.",

    "A photo of some people speaking.",
    "A photo of some people discussing.",
    "A photo of some people interacting.",
    "A photo of some people talking.",
    "A photo of some individuals connecting in a conversation.",
    "A photo of some people communicating during a discussion.",
    "A photo of some people working together on a project.",
    "A photo of some people speaking in a group setting.",
    "A photo of some people talking over a shared task.",
    "A photo of some individuals getting together for a collaboration.",
    "A photo of some people discussing ideas.",
    "A photo of some people coordinating on a project.",
    "A photo of some individuals working effectively together.",

    "A photo of various people speaking.",
    "A photo of various people discussing.",
    "A photo of various people interacting.",
    "A photo of various people talking.",
    "A photo of various individuals connecting in a team meeting.",
    "A photo of various people communicating during a session.",
    "A photo of various people working together on an assignment.",
    "A photo of various people speaking in a seminar.",
    "A photo of various people talking during a group activity.",
    # "A photo of various individuals getting together for a conference.",
    "A photo of various people discussing a collaborative project.",
    "A photo of various people coordinating on a common goal.",

    "A photo of a group of people speaking.",
    "A photo of a group of people discussing.",
    "A photo of a group of people interacting.",
    "A photo of a group of people talking.",
    "A photo of a group of people connecting in a workshop.",
    "A photo of a group of people communicating in a meeting.",
    "A photo of a group of people working together on a project.",
    # "A photo of a group of people speaking at a presentation.",
    "A photo of a group of people talking during a brainstorming session.",
    "A photo of a group of individuals getting together for a discussion.",
    "A photo of a group of people discussing their ideas.",
    "A photo of a group of people coordinating on a task.",
    "A photo of a group of individuals collaborating on a plan.",

    "A photo of many individuals interacting in a social setting.",
    # "A photo of multiple people interacting during a conference.",
    "A photo of several people interacting in a collaborative space.",
    "A photo of numerous persons interacting in a creative workshop.",
    "A photo of various individuals interacting during a team activity.",
    "A photo of some people interacting in a brainstorming session.",
    "A photo of a group of people interacting during a group exercise.",
    "A photo of a collective of people interacting in a meeting.",
    "A photo of a pair of individuals interacting over a project.",

    # "A photo of many people connecting during a networking event.",
    "A photo of multiple individuals connecting in a team setting.",
    # "A photo of several people connecting through a discussion.",
    # "A photo of numerous persons connecting at a workshop.",
    "A photo of various individuals connecting during a collaboration.",
    # "A photo of some people connecting during a team-building activity.",
    "A photo of a group of people connecting over a shared task.",
    "A photo of a collective of individuals connecting in a meeting.",
    # "A photo of a pair of people connecting through a collaborative effort.",

    # "A photo of many people communicating in a group meeting.",
    # "A photo of multiple individuals communicating during a workshop.",
    "A photo of several people communicating over a project.",
    # "A photo of numerous persons communicating in a conference room.",
    "A photo of various individuals communicating during a strategy session.",
    "A photo of some people communicating in a brainstorming group.",
    "A photo of a group of people communicating during a collaborative task.",
    "A photo of a collective of people communicating in a discussion.",
    "A photo of a pair of individuals communicating over a shared activity.",

    "A photo of many individuals working together on a project.",
    "A photo of multiple people working together",
    "A photo of several individuals working together",
    # "A photo of numerous persons working together in a team setting.",
    # "A photo of various individuals working together in a workshop.",
    "A photo of some people working together on a project.",
    # "A photo of a group of people working together effectively.",
    # "A photo of a collective of individuals working together towards a goal.",
    # "A photo of a pair of individuals working together on a task.",


    "A photo of many people speaking during a seminar.",
    "A photo of multiple individuals speaking in a group discussion.",
    # "A photo of several people speaking at a conference.",
    "A photo of numerous persons speaking during a presentation.",
    "A photo of various individuals speaking in a workshop.",
    "A photo of some people speaking at a team meeting.",
    "A photo of a group of people speaking in a brainstorming session.",
    "A photo of a collective of individuals speaking during a panel.",
    "A photo of a pair of individuals speaking during a discussion.",

    "A photo of many people talking during a collaborative meeting.",
    "A photo of multiple individuals talking in a group setting.",
    "A photo of several people talking during a brainstorming session.",
    "A photo of numerous persons talking in a team discussion.",
    "A photo of various individuals talking during a project review.",
    "A photo of some people talking in a casual work environment.",
    # "A photo of a group of people talking over a shared task.",
    # "A photo of a collective of individuals talking during a workshop.",
    # "A photo of a pair of people talking during a project planning",

    "A photo of many people getting together for a collaborative task.",
    "A photo of multiple individuals getting together for a meeting.",
    # "A photo of several people getting together for a brainstorming session.",
    # "A photo of numerous persons getting together in a workshop.",
    # "A photo of various individuals getting together for a team event.",
    "A photo of some people getting together for a project discussion.",
    # "A photo of a group of people getting together for a planning session.",
    # "A photo of a collective of individuals getting together for a presentation.",
    # "A photo of a pair of people getting together to work on a task.",

    "A photo of many individuals discussing a project.",
    # "A photo of multiple people discussing strategies in a meeting.",
    # "A photo of several individuals discussing ideas during a brainstorming session.",
    # "A photo of numerous persons discussing topics at a conference.",
    # "A photo of various individuals discussing plans in a group.",
    # "A photo of some people discussing solutions during a workshop.",
    # "A photo of a group of people discussing their findings.",
    "A photo of a collective of individuals discussing",
    # "A photo of a pair of individuals discussing their work together.",

    # "A photo of many people coordinating their efforts in a team.",
    "A photo of multiple individuals coordinating on a project.",
    "A photo of several people coordinating during a collaborative task.",
    # "A photo of numerous persons coordinating activities at an event.",
    "A photo of various individuals coordinating strategies in a meeting.",
    # "A photo of some people coordinating plans for a project.",
    # "A photo of a group of people coordinating their schedules.",
    # "A photo of a collective of individuals coordinating their actions.",
    # "A photo of a pair of individuals coordinating on a task.",

    "A photo of many individuals working collectively on a project.",
    "A photo of multiple people collaborating in a meeting.",
    "A photo of several individuals participating collectively in a discussion.",
    "A photo of numerous persons contributing collectively to a task.",
    # "A photo of various individuals acting collectively for a common goal.",
    # "A photo of some people working collectively in a team environment.",
    # "A photo of a group of people joining collectively for a cause.",
    # "A photo of a collective of individuals working collectively towards success.",
    # "A photo of a pair of individuals working collectively on a shared project.",


    "A photo capturing many faces.",
    "A photo featuring multiple faces.",
    "A photo capturing numerous faces.",
    "A photo of a group of faces.",
    "A photo highlighting some faces.",
    "A photo of a collective of faces.",
    "A photo depicting various faces.",

    "A photo of multiple people indoors.",
    "An image of several individuals gathered indoors.",
    "A picture showing a group of people indoors.",
    "A scene of multiple persons indoors.",
    "A depiction of a group of individuals indoors.",




    # "a photo of multiple people.",
    # "a photo of people interacting with each other.",
    # "a photo of people of different ages.",

    # "a photo of multiple faces.",

    # "a photo of multiple people standing.",
    # "a photo of multiple people sitting.",
    # "a photo of group of people standing.",
    # "a photo of group of people sitting.",

    # "a photo of a group taking a selfie.",
    # "a photo of friends posing for a selfie."
],
"single_person": [

    # "A photo of a single person working on a project.",
    # "A photo of a single individual engaged in a task.",
    # "A photo of a single person attending a meeting.",
    # "A photo of a single individual participating in an activity.",
    # "A photo of a single person working.",
    # "A photo of a single person in a busy workspace",

    "A photo of an individual working on a project.",
    "A photo of an individual engaged in a task.",
    "A photo of an individual attending a meeting.",
    "A photo of an individual participating in an activity.",
    "A photo of an individual working.",
    "A photo of an individual in a busy workspace",

    "A photo of a solo person working on a project.",
    "A photo of a solo individual engaged in a task.",
    "A photo of a solo person attending a meeting.",
    "A photo of a solo individual participating in an activity.",
    "A photo of a solo person working.",
    "A photo of a solo person in a busy workspace",

    "A photo of a sole person working on a project.",
    "A photo of a sole individual engaged in a task.",
    "A photo of a sole person attending a meeting.",
    "A photo of a sole individual participating in an activity.",
    "A photo of a sole person working.",
    "A photo of a sole person in a busy workspace",

    "A photo of a person alone working on a project.",
    "A photo of an individual alone engaged in a task.",
    "A photo of a person alone attending a meeting.",
    "A photo of a person alone participating in an activity.",
    "A photo of a person alone working.",
    "A photo of a person alone in a busy workspace",


    # "A photo of a person showing their face.",
    "An image featuring one person's face.",
    "A photo capturing an individual face.",
    "A photo capturing just one face.",

    # "A photo of a single face.",
    "A photo of an individual's face.",
    # "A photo of a face alone.",
    "A photo featuring an individual face.",
    # "A photo highlighting a single face.",
    # "A photo of an individual face in focus.",
    "A photo focused on an individual face.",
    "A photo containing a sole face.",
    "A photo of a solo face.", 

    # "A photo of a single person indoors.",
    "An image of an individual alone indoors.",
    "A picture showing a solitary person indoors.",
    # "A scene of a single person indoors.",
    "A depiction of an individual indoors.",


    # "a photo of a person.",

    # "a photo of a persons face.",
    # "a photo of a person standing.",
    # "a photo of a person sitting.",
    # "a photo of a person taking a selfie",

    # "a photo of a person working.",

],
"no_person": [

    "A photo of a place where nobody is present.",
    "A photo depicting a scene with nobody around.",
    # "A photo showing nobody in the frame.",
    "A photo that captures an area with nobody there.",
    
    "A photo of an empty room.",
    "A photo of an empty room where no one is present.",
    # "A photo showcasing a setting with no one around.",
    "A photo depicting a scene with no one in it.",
    "A photo capturing a moment with no one in view.",
    
    "A photo with no individuals present.",
    "A photo depicting a space absent of people.",
    "A photo of a scene that is absent of people.",
    "A photo capturing a moment that is absent of people.",
    
    "A photo depicting a scene devoid of individuals.",
    "A photo capturing a moment that is devoid of individuals.",
    
    # "A photo revealing an area empty of people.",
    
    "A photo of a scene lacking human presence.",
    "A photo depicting an area lacking human presence.",
    "A photo capturing a moment that is lacking human presence.",

    "A photo of nothing but an empty space.",
    "A photo of a still room without people.",
    "A photo devoid of people",
    "A photo that has no one in it",
    "A photo where no humans are visible",
    "A photo of a room with no people",

    "a photo of no visible faces.",
    "A photo with no faces present.",
    "A photo without any visible faces.",
    "A photo with no visible faces.",
    "A photo devoid of any human faces.",
    "A photo where no faces are visible.",
    "A photo capturing an area devoid of faces.",
    "A photo featuring a space without faces.",
    "A photo of a setting where no faces are visible.",

    "A photo of an empty indoor space.",
    "An image of a vacant indoor area.",
    "A picture showing an unoccupied indoor space.",
    "A scene of an empty indoor area.",
    "A depiction of an empty indoor space.",


    # "A photo of no people.",
    # "A photo of an empty room with no person visible",

    # "a photo of no visible faces.",

    # "A photo with no people",
    # "A photo that has no one in it",

    # "An image that does not contain any people",
    # "An image devoid of people",

    # "A picture showing an empty space",

    # "A picture where no humans are visible",
    # "A photo of an empty room with no people visible",
    # "An image showing an empty room with no people visible",

    # "A photo of a room with no people",
    # "An image showing a room with no person",

]}

# Load the CLIP model and processor
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")



# # Load an image
# image = Image.open("/tmp/video_incidents_ajeet/2578685/3_318.jpg")  # Replace with your image path

# # Preprocess the image
# inputs = processor_clip(images=image, return_tensors="pt")

# # Get the image embeddings
# with torch.no_grad():
#     outputs = model_clip.get_image_features(**inputs)

# # Normalize the embeddings
# image_embeddings = outputs / outputs.norm(dim=-1, keepdim=True)

# # Print the shape of the image embeddings
# print(image_embeddings.shape)

images, image_embeddings = clip_model.get_image_embeddings(["/home/ajeet/codework/dataset_frames/2529909/0_1218.jpg"])

precomputed_embeddings = clip_model.precompute_prompt_embeddings(text_prompts)

clip_label, clib_prob = clip_model.classify_image(image_embeddings[0], precomputed_embeddings)

print(clip_label)
print(clib_prob)