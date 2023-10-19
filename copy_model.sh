for date in 20231011-003229.166126 20231017-133304.123518
do
	sourceFolder=results/rescaled/$date
	mkdir -p $sourceFolder
	scp -r labgpu:~/scratchwork/anonymous/$sourceFolder/savedmodel ./$sourceFolder/
done
