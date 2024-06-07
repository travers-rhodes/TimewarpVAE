for date in 20231113-144159.904067 20231113-142737.784588
do
	sourceFolder=results/rateinvariantvae/$date
	mkdir -p ../$sourceFolder
	scp -r labgpu:~/scratchwork/iclr2024anonymousscratchwork/$sourceFolder/savedmodel ../$sourceFolder/
done
