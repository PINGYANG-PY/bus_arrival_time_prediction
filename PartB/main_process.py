
from reader import csv_reader
from preprocess import  preprocess
from model_select import  model_select
from plot_bar_chart import plot_bar_chart
from model_training import model_training
from test import test

path='Route24304Train.csv'

(x_date, x_vehicle, x_service, x_start, x_duration, x_spans,x_weekday_not)=csv_reader(path)
(x_weekday_not1,x_start1,x_stops1,x_travels1,x_date1, x_weekday_not2, x_stops2,x_travels2,x_start2,x_data2) = \
    preprocess(x_date, x_start, x_duration, x_spans, x_weekday_not)

(e_t_t, e_m_t, e_t_s, e_m_s, std_t_t, std_m_t, std_t_s, std_m_s) = model_select(x_weekday_not1,x_start1,x_stops1,x_travels1, x_date1)
plot_bar_chart(e_m_t, std_m_t)
plot_bar_chart(e_m_s, std_m_s)
plot_bar_chart(e_t_t, std_t_t)
plot_bar_chart(e_t_s, std_t_s)

model_training(x_weekday_not1,x_start1,x_stops1,x_travels1, x_date1)
(MAE_travel_left, MAE_stops_left, MAE_travel_top, MAE_stop_top, MAE_stop_top, MAE_travel_main)\
    =test(x_weekday_not2, x_start2, x_stops1, x_travels2, x_date2)


