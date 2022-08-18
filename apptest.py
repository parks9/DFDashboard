#Import all the useful packages

import streamlit as st
import pandas as pd
import datetime
import numpy as np
from summ import basic_info, frame_per_lens, flag_count, total_flags
import matplotlib.pyplot as plt
import dfreduce



#Trial data for the app test
from DataBase600.upload import data

lightFlags, darkFlags, flatFlags = dfreduce.flags.LightFlags.bit_to_str, dfreduce.flags.DarkFlags.bit_to_str, \
                                    dfreduce.flags.FlatFlags.bit_to_str


#Writing a quick title for the page

st.write("""
# The Dragonfly Telephoto Array
*Ultra-low surface brightness astronomy at visible wavelengths*


This will be the dashboard where the Dragonfly data is displayed and orgranized.

""")




#Setting a couple datetime variables that might be useful later

yesterday = datetime.date.today() - datetime.timedelta(days=1)




#Setting a 'night' variable which is an input from the user // Defaults to a night with available frames

night = st.date_input("Night of observation", value=datetime.date(2022, 4, 8), max_value= datetime.date.today())




# Alerting user that the data is available or not

if str(night) not in data['date'].values:
    st.write('Night not available.')
    #st.experimental_rerun()
#     st.error('Date not available')
else:
    st.write('Data from the night of', night, 'is available.')
    
    
    

# Filtering the data that appears on the chosen night
    
is_there = data['date']== str(night)
     
don = data[is_there] #DON means Data Of Night




# Calling the "basic_info" function to display the number of frames on chosen night
# and the 'quality' of the frames // See the Summ package for more info

a,b,c,d,e,f = basic_info(don)

st.write('There are', a, 'light frames,', b, 'dark frames, and', c, 'flat frames.')



st.write(d, '% of the light frames are good.')

st.write(e, '% of the dark frames are good.')

st.write(f, '% of the flat frames are good.')




# A button that user can click to see the data on the chosen night // Display style >> Dataframe

if st.button('Click to show data'):
    st.dataframe(don)


    
    
# The goal here is to separate the data based in date and store the 4 days
# prior to the selected date in a list

group_date = data.groupby('date')

dates_listed = np.array(list(group_date.groups.keys()))

index_date = np.where(dates_listed == str(night))

#st.write(index_date[0][0])


days_c = []
for i in reversed(range(6)):
    days_c.append(dates_listed[index_date[0][0] - i])

adj_info10 = []
adj_info20 = []
adj_info30 = []

adj_q10 =[]
adj_q20 = []
adj_q30 = []

for j in days_c:
    li, dk, fl, ql, qd, qf = basic_info(data[data['date'] == j])
    
    adj_info10.append(li)
    adj_info20.append(dk)
    adj_info30.append(fl)
    
    adj_q10.append(ql)
    adj_q20.append(qd)
    adj_q30.append(qf)
    

adj_info1 = np.nan_to_num(adj_info10)
adj_info2 = np.nan_to_num(adj_info20)
adj_info3 = np.nan_to_num(adj_info30)
adj_q1 = np.nan_to_num(adj_q10)
adj_q2 = np.nan_to_num(adj_q20)
adj_q3 = np.nan_to_num(adj_q30)




# Making two buttons that will display the previous info in plots
    
if st.button("Show number of frames taken 5 days before chosen date"):    
    figure = plt.figure(figsize=(12,11),tight_layout=True)
    plt.style.use('dark_background')
    plt.plot(days_c, adj_info1, label='Light Frames')
    plt.plot(adj_info2, label='Dark Frames')
    plt.plot(adj_info3, label='Flat Frames')

    plt.legend()


    plt.title('Number of Frames Taken on 5 Previous Days', size=15)
    plt.ylabel('Number of Frames', size=13)
    plt.xlabel('Day', size=13)
    
    #plt.show()

    st.pyplot(figure)
    
    
if st.button("Show quality of images taken 5 days before chosen date"):
    figure = plt.figure(figsize=(8.5,6),tight_layout=True)
    plt.style.use('dark_background')
    plt.plot(days_c, adj_q1, label='Light Frames')
    plt.plot(adj_q2, label='Dark Frames')
    plt.plot(adj_q3, label='Flat Frames')

    plt.legend()


    plt.title('Quality of Frames Taken on 5 Previous Days', size=20)
    plt.ylabel('Number of Good Frames (%)', size=20)
    plt.xlabel('Day', size=20)


    #plt.show()

    st.pyplot(figure)

    
# Quick display of the basic info separated by camera    
    
i,j = frame_per_lens(don)
    
    
st.markdown("# Break down of frames per camera and their quality")
    
st.dataframe(j, width=2000)


###DECODING FLAGS

#First let's just get a list of all the flags that can show up in each type of frame

LF, DF, FF = list(lightFlags.values()), list(darkFlags.values()), list(flatFlags.values())

flag_names = list(lightFlags.values()), list(darkFlags.values()), list(flatFlags.values())


# If a frame has flags, store the flag int in a list which we will later convert to binary or string and count 
# the number of occurences for each flag

# flagged = don[don['flags'] != 0]

# lightF, darkF, flatF = flagged[flagged['frame_type'] == 'light'], flagged[flagged['frame_type'] == 'dark'], \
#                         flagged[flagged['frame_type'] == 'flat']

# further = list(darkF['flags'])
# fur2 = np.zeros(len(DF))


# for i,j in enumerate(further):
#     k =  np.binary_repr(j, width=10)
#     vals = np.array([int(a) for a in k])
    
#     fur2 += vals

all_flags = total_flags(don, flag_names)
    

#n, bins, _ = plt.hist(fur2, 10, density=False, facecolor='r')
#plt.xticks(np.arange(len(DF)), labels=DF)




#for i in range(len(further[0])):
#    histo_list.append(np.sum(further[:][i]))

scale = 25
width = 0.4
title_size = 15
axis_size = 10

fig = plt.figure()

x = np.arange(len(DF))

DF = [i.replace('_', ' ') for i in DF]

plt.title('Flag Occurences in Dark Frames', size=title_size)

plt.bar(x, all_flags[1], width, label='Ultra Diffuse')
plt.xlabel('Name of flag', size=axis_size)
plt.ylabel('Number of each flag', size=axis_size)

plt.tick_params(axis='y', labelsize=20)
plt.xticks(np.flip(x), labels=DF, rotation=80, size=7)

plt.tight_layout()
#plt.show()

st.pyplot(fig)






fig2 = plt.figure()

x2 = np.arange(len(FF))

FF = [i.replace('_', ' ') for i in FF]

plt.title('Flag Occurences in Flat Frames', size=title_size)

plt.bar(x2, all_flags[2], width, label='Ultra Diffuse')
plt.xlabel('Name of flag', size=axis_size)
plt.ylabel('Number of each flag', size=axis_size)

plt.tick_params(axis='y', labelsize=20)
plt.xticks(np.flip(x2), labels=FF, rotation=80, size=7)

plt.tight_layout()
#plt.show()

st.pyplot(fig2)




fig3 = plt.figure()

x3 = np.arange(len(LF))

LF = [i.replace('_', ' ') for i in LF]

plt.title('Flag Occurences in Light Frames', size=title_size)

plt.bar(x3, all_flags[0], width, label='Ultra Diffuse')
plt.xlabel('Name of flag', size=axis_size)
plt.ylabel('Number of each flag', size=axis_size)

plt.tick_params(axis='y', labelsize=20)
plt.xticks(np.flip(x3), labels=LF, rotation=80, size=7)

plt.tight_layout()
#plt.show()

st.pyplot(fig3)

st.write(all_flags[1])




    
    





##NOTES TO WORK ON

#Check to see if you can use DFIndividualFrames to do this instead of importing 



