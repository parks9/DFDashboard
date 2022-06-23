import streamlit as st
import pandas as pd
import datetime
import numpy as np
from summ import basic_info, frame_per_lens
import matplotlib.pyplot as plt


#Trial data for the app test
from DataBase600.upload import data


st.write("""
# The Dragonfly Telephoto Array
*Ultra-low surface brightness astronomy at visible wavelengths*


This will be the dashboard where the Dragonfly data is displayed and orgranized.

""")



yesterday = datetime.date.today() - datetime.timedelta(days=1)

night = st.date_input("Night of observation", value=datetime.date(2022, 6, 14), max_value= datetime.date.today())

if str(night) not in data['date'].values:
    st.write('Night not available.')
    #st.experimental_rerun()
#     st.error('Date not available')
else:
    st.write('Data from the night of', night, 'is available.')
    
is_there = data['date']== str(night)
     
don = data[is_there] #DON means Data Of Night


a,b,c,d,e,f = basic_info(don)

st.write('There are', a, 'light frames,', b, 'dark frames, and', c, 'flat frames.')



st.write(d, '% of the light frames are good.')

st.write(e, '% of the dark frames are good.')

st.write(f, '% of the flat frames are good.')


if st.button('Click to show data'):
    st.dataframe(don)

    
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


    
    
    
if st.button("Show number of frames taken 5 days before chosen date"):    
    figure = plt.figure(figsize=(8.5,6),tight_layout=True)
    plt.style.use('dark_background')
    plt.plot(days_c, adj_info1, label='Light Frames')
    plt.plot(adj_info2, label='Dark Frames')
    plt.plot(adj_info3, label='Flat Frames')

    plt.legend()


    plt.title('Number of Frames Taken on 5 Previous Days', size=20)
    plt.ylabel('Number of Frames', size=20)
    plt.xlabel('Day', size=20)
    
    plt.show()

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


    plt.show()

    st.pyplot(figure)

i,j = frame_per_lens(don)
    
    
st.markdown("# Break down of frames per camera and their quality")
    
st.dataframe(j, width=2000)

    
    
    









