#for i in `seq 11 32`
#python3 pizdaint_full_analysis.py $i spike1s --gpu0 &
#python3 pizdaint_full_analysis.py $i spike1s_gaba --gpu1 &

#013 2021-12-08-shadow_averages_0.001_0.2_5e889db2-932e-4e73-ba07-0772e4fa6a4a
#024 2021-12-08-shadow_averages_0.01_0.2_5e889db2-932e-4e73-ba07-0772e4fa6a4a

#019 2021-12-08-shadow_averages_0.001_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be
#030 2021-12-08-shadow_averages_0.01_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be
#052 2021-12-08-shadow_averages_1_0.8_d1666304-c6fc-4346-a55d-a99b3aad55be


#python3 pizdaint_full_analysis.py 19 comp_figure2c_gaba_v2 --gpu0 &
#python3 pizdaint_full_analysis.py 30 comp_figure2c_gaba_v2 --gpu1 &
#wait

#
#python3 pizdaint_full_analysis.py 19 comp_ghalf --gpu0 &
#python3 pizdaint_full_analysis.py 52 comp_ghalf --gpu1 &
#wait
#python3 pizdaint_full_analysis.py 19 comp_spike1target_gaba_v2_probe --gpu0 &
python3 pizdaint_full_analysis.py 30 comp_figure2c_gaba_v2_probe --gpu0 &
python3 pizdaint_full_analysis.py 30 comp_spike1target_gaba_v2_probe --gpu1 &

wait
#python3 pizdaint_full_analysis.py 19 comp_spike1full_gaba_v2 --gpu0 &
#python3 pizdaint_full_analysis.py 30 comp_spike1full_gaba_v2 --gpu1 &
