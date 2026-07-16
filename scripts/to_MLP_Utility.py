import torchaudio
import os
import matplotlib.pyplot as plt
import math
import torchaudio
import torch
import math


#numpy array is fast for math but for vectorized operations (math on whole arrays at once) 
#in append, numpy recreate the entire array instead of adding the latest tail. for append, better to use list.
#Gather data → use Python lists.
#Do math / transformations → use NumPy arrays.
def plotter(datasets):
    chart_data_name = []
    chart_data_populus = []

    for i, (name, (file_paths, files)) in enumerate(datasets.items(), start=1):
        #just match the shape of the dictionary here (key, (value1, value2))
        chart_data_populus.append(len(files))
        chart_data_name.append(name)
        print(f"jumlah audio {chart_data_name[i-1]}: {chart_data_populus[i-1]}")
        print("Example:")
        for j, f in enumerate(files[:5], start = 1): #loop for displayiing 5 samples
        # j is index, f is files    
            print(f"  {j}. {f}")
            #shape of waveform tensor per samples.
            waveform_path = os.path.join(file_paths, f)
            waveform, sample_rate = torchaudio.load(waveform_path)  
            print(waveform.shape, sample_rate)


        if i % 2 == 0:
            sizes = [chart_data_populus[i-2], chart_data_populus[i-1]]
            labels = [chart_data_name[i-2], chart_data_name[i-1]]

            plt.figure(figsize=(5,5))
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.axis("equal")   # circle shape
            plt.show()
        print("-" * 40)

    # f"{name}" --> f before string: "formatted string"
    # Inside the string, anything in {} is evaluated as Python code.
    #another example with no formatted string:
    #print("jumlah audio", name, ":", len(files))


    #enumerate() is a Python built-in function that adds a 
    #counter to an iterable (like a list, dict, etc.) 
    #so you can loop over both the items and their 
    #index at the same time.

    # f"{name}" --> f before string: "formatted string"
    # Inside the string, anything in {} is evaluated as Python code.
    #another example with no formatted string:
    #print("jumlah audio", name, ":", len(files))


    #enumerate() is a Python built-in function that adds a 
    #counter to an iterable (like a list, dict, etc.) 
    #so you can loop over both the items and their 
    #index at the same time.
    sizes = chart_data_populus
    labels = chart_data_name

    plt.figure(figsize=(5,5))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")   # circle shape
    plt.show()


def audio_slicer(treshold_high, treshold_low, data_set, data_set_name):
    sliced_audio = []

    for i, items in data_set.iterrows():
        waveform, sample_rate = torchaudio.load(data_set.loc[i, 'file path'])
        
        max_samples = int(sample_rate * treshold_high)
        min_samples = int(sample_rate * treshold_low)

        print("audio will be cut to this rate:", max_samples)
        waveform_size = waveform.size(1) #--> get the second column of waveform.size() which is the number of samples in each waveform
        print(f"this audio's duration: {data_set.loc[i, "duration"]}")
        print(f"samples: {waveform_size}")

        chunks = [] #--> nanti isinya audio pendek pendek yg udh di cut per audio panjang

        num_chunks = math.ceil(waveform_size / max_samples)
        #Calculates how many chunks we need to cover the whole audio.
        #ceil rounds up, so the last chunk is included even if smaller than chunk_size.

        for i in range(num_chunks):
            #where the program start and end the cutting
            start = i * max_samples
            end = start + max_samples
            chunk = waveform[:, start:end] #--> slicing the audio. keep in mind to use waveform because waveform is the actual audio (because it has channel and samples)
            chunk_size = chunk.size(1)

            # [:] keep the channel, cut the samples from start to end
            print(f"this chunk's size is {chunk_size}")
            if chunk_size > max_samples: #if above treshold:
                #cut to treshold, make it a new file
                chunk = chunk[:, :max_samples]
                chunks.append(chunk)
                print(f"chunk above max treshold, slicing.")
            elif chunk_size < min_samples: #if below treshold
                #erase/trow away
                print(f"chunk below min treshold, skipping.")
                continue  
                #continue is to skip    
            else:
                #if within treshold, pad to the max samples
                pad_size = max_samples - chunk.size(1)
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                chunks.append(chunk)
                print(f"chunk within treshold, padding.")
        sliced_audio.append(chunks)
        print("-"*40)
        
    #checker
    checker = []
    for e, slices in enumerate(sliced_audio):
        sample_sliced = sliced_audio[e]
        checker.append(any(i.size(1) != max_samples for i in sample_sliced)) #check if there exist any non 96000 in a list
    print(f"still {treshold_high} seconds in{data_set_name}:  {all(i == False for i in checker)}")
    return sliced_audio

def flatten_waveform(sample):
    # Unwrap nested lists
    while isinstance(sample, list):
        sample = sample[0]
    return sample

#gpu

def feature_extractor_gpu(sample_sequence, model_size="base", layer_index=None):
    # pick device
    device = torch.device("cuda")

    # choose wav2vec bundle correctly
    if model_size == "base":
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
    else:
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE

    # load model and send to correct device (GPU if available)
    extractor_model = bundle.get_model().to(device)
    extractor_model.eval()

    print(f"Using model: WAV2VEC2_{model_size.upper()} on {device}")
    print("Expected sample rate:", bundle.sample_rate)
    print(f"Starting extraction for {len(sample_sequence)} samples...\n")

    features = []
    total = len(sample_sequence)

    for idx, sample in enumerate(sample_sequence, start=1):
        waveform = flatten_waveform(sample)

        # convert python list to tensor
        if isinstance(waveform, list):
            waveform = torch.tensor(waveform)

        # ensure [1, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # move input tensor to same device as model
        waveform = waveform.to(device)

        with torch.inference_mode():
            # get hidden layers
            hidden_states, _ = extractor_model.extract_features(waveform)

        # pick layer(s)
        if layer_index is None:
            # detach from gpu and keep on cpu for storage
            feature = [h.detach().cpu() for h in hidden_states]
        else:
            feature = hidden_states[layer_index].detach().cpu()

        features.append(feature)

        print(f"[{idx}/{total}] OK — waveform shape {tuple(waveform.shape)}")

    print("\nFeature extraction completed.")
    return features



#cpu
def feature_extractor(sample_sequence, model_size="base", layer_index=None):
    # Select model
    if model_size == "base":
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
    else:
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE
    
    extractor_model = bundle.get_model()
    extractor_model.eval()

    print(f"Using model: WAV2VEC2_{model_size.upper()}")
    print("Expected sample rate:", bundle.sample_rate)
    print(f"Starting extraction for {len(sample_sequence)} samples...\n")

    features = []
    total = len(sample_sequence)

    for idx, sample in enumerate(sample_sequence, start=1):
        waveform = flatten_waveform(sample)

        # Convert list → tensor if needed
        if isinstance(waveform, list):
            waveform = torch.tensor(waveform)

        # Ensure shape: [1, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        with torch.inference_mode():
            # Extract all hidden layers
            # This returns: (list_of_hidden_states, output_dict)
            hidden_states, _ = extractor_model.extract_features(waveform)

        # Decide what to save
        if layer_index is None:
            # Save ALL layers (list of tensors)
            feature = hidden_states
        else:
            # Save ONE layer (tensor)
            feature = hidden_states[layer_index]

        features.append(feature)

        print(f"[{idx}/{total}] OK — waveform shape {tuple(waveform.shape)}")

    print("\nFeature extraction completed.")
    return features



'''

to_MLP_Utility.py defines several functions that are crucial in the process done in preprocessing.ipynb

1. plotter()
the plotter() function gives a basic dataset reading to understand the number of samples and its
distributions in the form of a pie chart. It also shows the five first samples in the dataset.

2. audio_slicer()
the audio_slicer() function takes a list of paths for the audio dataset, load it as waveform, and slice it
into chunks based on its duration. audio with duration below low_treshold will be skipped. audio with duration
in between low_treshold and high_treshold will be padded, and audio with duration above high_treshold will be sliced
into chunks and its tail skipped if below low_treshold and padded if follows the second condition instead.

3. feature extractors
There are two feature extractor functions in the module. feature_extractor() and feature_extractor_gpu(). Both
are identical in its function but gpu integration. these functions takes the sample sequence of the audio data,
select a model size (base or large), select the extraction layer to be used, and recursively extract features in 
all the samples in the sample sequence list.

'''