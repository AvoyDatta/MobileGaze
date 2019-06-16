##MobileGaze: An efficient framework for mobile gaze tracking##

The MobileGaze paper can be found in the folder MobileGaze/ee267w_submissions/ .

Author: Avoy Datta (BSc. Stanford EE, 2020). 

To setup project:

1. Download GazeCapture dataset from https://github.com/CSAILVision/GazeCapture. This downloads a .tar archive. Store the .tar archive in the "data" folder.
2. Run MobileGaze/data/extract.sh to extract data into MobileGaze/data/.
3. Create an empty folder MobileGaze/processed/ . Run the preprocessing script provided at the dataset website to process the data in MobileGaze/data/ and store the processed frames in MobileGaze/processed/ 
4. Acquire the saved checkpoints for the model, saving the .pth.tar files in MobileGaze/src/saved_models/sn/
5. To test MobileGaze trained for 5 epochs, run the shell script MobileGaze/src/test_sn.sh. This uses the checkpoint with path saved_models/sn/best_checkpoint.pth.tar. This should give the results provided in the MobileGaze paper. 

Contact author at avoy.datta@stanford.edu for further queries. 
