# Motion-Based Verification Prototype
This folder includes all the necessary files to run the Unity prototype with a VR headset (we used the Oculus Quest 2).

## Setup Instructions
Before you can use this Unity prototype, you need to set up your environment:

1. Install `Unity Version 2022.3.25f1`.
2. Open the project in Unity.
3. Add the scene from `Assets/Scenes/Motion-Password` to the Unity hierarchy.
4. Connect the VR headset to the PC.

## Start Prototype
The `motion-password-server` must be started before the Unity scene is started.
If the VR headset is properly connected to the PC, simply start the Unity scene and the application will work.

## Usage Instructions

There are two modes: registration and verification mode.

### 1. Registration Mode

Here you can register as a new user:

1. click on "Create New User" on the left board
2. hit red button in front of you to start recording
3. use either hand to create signature; press "grip" button on controller to 'paint'
4. hit red button in front of you again to stop recording
5. repeat this a few times with the same signature; try to be consistent for better results
   
### 2. Verification Mode

Switch to verification mode by clicking on the "Verify"-tab on the left board

1. select one of the identities from the left board
2. enter your signature as before with the red button in front of you
3. after you entered a signature, the result will be shown after a few seconds on the right board: green for success, red for access denied; you can also see some debugging information, such as the achieved cosine similarity score (see paper for details)
4. ideally, the system accepts your own signatures, and rejects if you try to login as someone else ;-)
5. you can also choose one of the four thresholds we selected in the paper (from 'strict' to 'permissive'), which controls how lenient the verification system is

