# Dragonfly Telescope Dashboard Project

## File purposes:

- apptest.py: contains all of the code pretaining to the Streamlit app. You must have the Streamlit python
package installed, as well as the dfreduce package from the private dragonfly repo.

- requirements.txt: contains the additionnal package installations that streamlit must do to compile the 
apptest.py code. Note that we have not yet found a way to get Streamlit to install private Github repo content.
Streamlit deploys the dashboard app from this git repo and reads this file before the main app script.

- summ.py: contains all of the functions used to reduce the data for all the current features (e.g. compiling
the flags, finding the number of frames...)


## What's next

- We would like to connect the app to the current data coming from the telescope, which gets stored in S3.
- Adding various features, depending on what the dragonfly team thinks would be useful.
- Hosting the dashboard such that a team member can access it anytime from anywhere.


