'''
helpers.py
By Lance Mathias
Helper functions for calculating the sharpness of waveforms using a 
lowpass filter and helpers from the 16b notebook to allow the code
to stand alone.
'''

import numpy as np
import matplotlib.pyplot as plt

def make_filter(size: int, mu: float, sigma: float) -> np.array:
    '''
    Creates a 1-d lowpass Gausian filter with the given parameters.

        Parameters:
            size (int): The "width" of the Gausian filter. 
            mu (float): The mean of the Gausian distribution.
            sigma (float): The standard deviation of the Gausian filter. Controls how much more values around the
            mean affect the filter.

        Returns:
            A Numpy array corresponding to the given Gausian filter.
    '''
    return np.array([(2.*np.pi*sigma**2. )**-.5 * np.exp( -.5 * (x-mu)**2. / sigma**2. ) for x in range(size)])

def calc_sharpness(s: np.array, lowpass: np.array) -> float:
    '''
    Calculates the "sharpness" of the given sample by running the specified lowpass filter on the signal and 
    calculating the total squared error of the original signal compared to the smoothed signal. Lower values
    mean the signal is closer to smooth, while higher values mean the signal has lots of spikes.

        Parameters:
            s (np.ndarray): The 1-d array of the signal to be analyzed.
            lowpass (np.ndarray): The filter to use to smooth the signal.

        Returns:
            A float value corresponding to the calculated "sharpness" of the signal.
    '''
    error = s - np.convolve(s, lowpass, 'same') 
    return np.sqrt(np.sum(np.square(error))/len(s))

# Helpers from the eecs16b repository
def get_snippets(data, length, pre_length, threshold):
    """Attempts to align audio samples in data.
    
    Args:
        data (np.ndarray): Matrix where each row corresponds to a recording's audio samples.
        length (int): The length of each aligned audio snippet.
        pre_length (int): The number of samples to include before the threshold is first crossed.
        threshold (float): Used to find the start of the speech command. The speech command begins where the
            magnitude of the audio sample is greater than (threshold * max(samples)).
    
    Returns:
        Matrix of aligned recordings.
    """
    assert isinstance(data, np.ndarray) and len(data.shape) == 2, "'data' must be a 2D matrix"
    assert isinstance(length, int) and length > 0, "'length' of snippet must be an integer greater than 0"
    assert 0 <= threshold <= 1, "'threshold' must be between 0 and 1"
    snippets = []

    # Iterate over the rows in data
    for recording in data:
        # Find the threshold
        recording_threshold = threshold * np.max(recording)

        # Figure out when interesting snippet starts
        i = pre_length
        while recording[i] < recording_threshold:
            i += 1
            
        snippet_start = min(i - pre_length, len(recording) - length)
        snippet = recording[snippet_start:snippet_start + length]

        # Normalization
        snippet = snippet / np.sum(snippet)
        
        snippets.append(snippet)

    return np.vstack(snippets)

# Wrapper function for get_snippets
def process_data(dict_raw, length, pre_length, threshold, plot=True):
    """
    Process the raw data given parameters and return it.
    
    Args:
        dict_raw (np.ndarray): Raw data collected.
        data (np.ndarray): Matrix where each row corresponds to a recording's audio samples.
        length (int): The length of each aligned audio snippet.
        pre_length (int): The number of samples to include before the threshold is first crossed.
        threshold (float): Used to find the start of the speech command. The speech command begins where the
            magnitude of the audio sample is greater than (threshold * max(samples)).
        plot (boolean): Plot the dataset if true.
            
    Returns:
        Processed data dictionary.
    """
    processed_dict = {}
    word_number = 0
    for key, word_raw in dict_raw.items():
        word_processed = get_snippets(word_raw, length, pre_length, threshold)
        processed_dict[key] = word_processed
        if plot:
            plt.plot(word_processed.T)
            plt.title('Samples for "{}"'.format(selected_words_arr[word_number]))
            word_number += 1
            plt.show()
            
    return processed_dict 