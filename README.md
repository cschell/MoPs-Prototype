# Motion-Based Verification Prototype

To demonstrate the feasibility of motion-based verification in XR environments, we developed a prototype Unity application.
This prototype integrates our trained similarity-learning model, as used in the analyses presented in our paper.
To our knowledge, this represents the first published application incorporating motion-based recognition in an XR setting.

There are two components:
- **Python server**: The server manages user data and handles the registration and verification of motion sequences through an HTTP API.  It executes the similarity-learning model and processes the motion data accordingly.
- **Unity scene**: The Unity scene provides the virtual environment and acts as a client to communicate with the Python server. This repository contains two components: a Unity project with a prototype XR user interface and a Python server that hosts the motion-based verification model.

For detailed setup instructions, please refer to the README files in each respective directory.
