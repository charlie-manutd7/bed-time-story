Here are the steps needed to set up the virtual environment:
Step 1: Create a virtual environment
python3 -m venv myenv
Step 2: Activate the virtual environment
source myenv/bin/activate
Step 3: Install necessary packages
pip install -r requirements.txt

There's also another folder that contains voice sample. It's not included in this repository, but I can share with you if needed. 

I was able to go through executing all the main steps:
- run prepare_data.py
- run train.py
- run inference.py

I was able to generate an output.wav, but it was empty. ChatGPT-4o suggested that it was probably because the inference wasn't run based on any checkpoint after training the data, so it wasn't able to generate any meaningful output. 
That's why I went back to change the code to make sure it saves a checkpoint, and the inference code and use that checkpoint. 
  
However, I'm running into the problem that the code failed to save and create a checkpoint after running train.py. No error was seen, but the output was just a bunch of debug codes without any meaningful log. 
Plus, the checkpoints directory is empty, while it should contain path files to the checkpoint after each time I train the model. So I suspect that the code wasn't successfully executed.
ChatGPT-4o wasn't able to help in debugging why that happened. So I'm stuck here.
