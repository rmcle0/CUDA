C_CPP_FLAGS = -O3

all: default consequential thrust 

launch: default consequential thrust 
	mpisubmit.pl default 
	mpisubmit.pl consequential
	bsub -oo "thrust.txt" -gpu "num=1:mode=exclusive_process" ./thrust


default: jac3d.c 
	gcc jac3d.c $(C_CPP_FLAGS) -o default 

consequential: jac3d.cpp
	g++ -std=c++11  $(C_CPP_FLAGS) jac3d.cpp -o consequential

thrust: thrust.cu 
	nvcc -std=c++11 $(C_CPP_FLAGS) thrust.cu -o thrust

clean:
	rm -f *.dump *.err *.out .*o *.exe
	
send:
	scp -i /home/sdev/.ssh/id_rsa_hpc   *.c *.cpp *.cu  Makefile edu-cmc-skmodel24-607-01@polus.hpc.cs.msu.ru:/home_edu/edu-cmc-skmodel24-607/edu-cmc-skmodel24-607-01/CUDA1


compare:
	diff consequential.dump thrust.dump

print:
	clear
	cat test*.out
	cat test*.err
