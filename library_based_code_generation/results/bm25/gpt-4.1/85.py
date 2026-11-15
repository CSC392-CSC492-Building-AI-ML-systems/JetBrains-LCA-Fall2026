```python
from expyfun import ExperimentController, screen_prompt, screen_text, plot_screen
from expyfun.stimuli import crm_prepare_corpus, crm_sentence, CRMPreload, crm_response_menu
import numpy as np
import time

# Experiment parameters
exp_name = 'CRM_Experiment'
window_size = (1024, 768)
participant = 'P001'
session = 'S001'
version = 'v1'
fs = 40000  # Sampling rate
talker_num = 1
duration = 2.0  # seconds to play stimulus

# Prepare CRM corpus for two talkers (male and female, same talker number)
corpus = crm_prepare_corpus(fs=fs)
valid_callsigns = corpus['callsigns']
print("Valid callsigns:", valid_callsigns)

# Prepare two talkers
talker_male = crm_sentence(callsign='Arrow', color='red', number='1', talker='male', talker_num=talker_num, fs=fs)
talker_female = crm_sentence(callsign='Arrow', color='red', number='1', talker='female', talker_num=talker_num, fs=fs)

# Read a sentence from hard drive (simulate by using crm_sentence)
sentence1 = crm_sentence(callsign='Arrow', color='red', number='1', talker='male', talker_num=talker_num, fs=fs)

# Preload all talkers
preloader = CRMPreload(fs=fs)
preloader.preload_all()

# Get a second sentence from memory
sentence2 = preloader.get_sentence(callsign='Arrow', color='red', number='1', talker='female', talker_num=talker_num)

# Pad and align sentences at the start
max_len = max(len(sentence1), len(sentence2))
padded1 = np.pad(sentence1, (0, max_len - len(sentence1)), mode='constant')
padded2 = np.pad(sentence2, (0, max_len - len(sentence2)), mode='constant')
stims = np.stack([padded1, padded2])

# Run experiment
with ExperimentController(exp_name, window_size=window_size, participant=participant,
                         session=session, version=version, full_screen=False, audio_fs=fs) as ec:
    # Display text prompt
    screen_text(ec, "Listen to the sentences and respond.", wrap=True)
    ec.flip()
    time.sleep(1.0)

    # Load padded sentences into buffer
    ec.load_buffer(stims)

    # Start stimulus
    ec.start_stimulus()
    ec.wait_secs(duration)

    # Response menu
    response, rt = crm_response_menu(ec, valid_callsigns, colors=corpus['colors'], numbers=corpus['numbers'],
                                     prompt='Select the correct sentence:', wrap=True)
    # Display prompt based on response
    if response is not None:
        prompt = f"You selected: {response}"
    else:
        prompt = "No response detected."
    screen_text(ec, prompt, wrap=True)
    ec.flip()
    time.sleep(1.0)

    # Plot screenshot of the experiment
    plot_screen(ec)
```