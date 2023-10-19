rm paper_images/*.png
rm paper_images/*.pdf
rm paper_images/*.tex

for notebookName in ForkTrainingDataVisualization.ipynb AblationPlot.ipynb ExampleLetterA.ipynb InterpolateTwoAs-NNModel.ipynb ForkTrainingDataVisualization.ipynb TimingNoisePlot.ipynb BaselineMethodsInterpolateTwoAs.ipynb VisualizationLatentSweep.ipynb dmpInterpolateA.ipynb ForkTrainingResultsTable.ipynb

do
	jupyter nbconvert --execute --to notebook paper_images/$notebookName
done

