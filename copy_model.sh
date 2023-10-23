for date in 20231011-003229.166126 20231022-200601.075614 20231020-163255.573136 20231021-232100.355514
do
	sourceFolder=results/rescaled/$date
	mkdir -p $sourceFolder
	scp -r labgpu:~/scratchwork/anonymous/$sourceFolder/savedmodel ./$sourceFolder/
done
