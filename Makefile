all: | forkdata/forkTrajectoryData.npz data/trainTest2DLetterACache.npz

forkdata/forkTrajectoryData.npz: | forkdata/20230222_194413/tip_pose_45.npy
	jupyter nbconvert --execute --to notebook --inplace forkdata_formatting/CleanAndRandomlySampleForkData.ipynb

forkdata/20230222_194413/tip_pose_45.npy: | forkdata/fork_trajectory_recordings.zip
	unzip forkdata/fork_trajectory_recordings.zip -d forkdata/

forkdata/fork_trajectory_recordings.zip:
	mkdir -p forkdata
	cp raw_fork_trajectory_data/fork_trajectory_recordings.zip forkdata/

data/trainTest2DLetterACache.npz : data/trainTest2DLetterARescaled.npz
	./data_formatting/cache_noise_added_A.py

data/trainTest2DLetterARescaled.npz : data/combined_trajectories.pickle data_formatting/test_train_A.py
	./data_formatting/test_train_A.py 

data/combined_trajectories.pickle: | data/matR_char_numpy/lower_a_C1_t01.npz
	./data_formatting/combine_python_data.py

data/matR_char_numpy/lower_a_C1_t01.npz: | data/matR_char/lower_a_C1_t01.mat
	mkdir -p data/matR_char_numpy
	./data_formatting/convert_to_python.py
	mv data/matR_char/*.npz data/matR_char_numpy/

data/matR_char/lower_a_C1_t01.mat: | data/matR_char_112712.rar
	mkdir -p data/matR_char
	unrar e data/matR_char_112712.rar data/matR_char/

data/matR_char_112712.rar:
	mkdir -p data
	wget -O data/matR_char_112712.rar https://www.dropbox.com/s/6krfe2i23llnwvl/matR_char_112712.rar?dl=0

clean:
	rm -rf data/*
	rm -rf forkdata/*
