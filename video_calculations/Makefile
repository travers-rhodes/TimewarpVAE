filename_template = latent$(lat)-$(azim_elev).mp4
convert_video=ffmpeg -framerate 20 -i animatedLatent/latent$(lat)-$(azim_elev)_frame%d.png latent$(lat)-$(azim_elev).mp4
reversed_video=ffmpeg -framerate 20 -i animatedLatent/latent$(lat)-$(azim_elev)_frame%d.png -vf reverse latent$(lat)-$(azim_elev)-reversed.mp4
lats := 0 1 2
azim_elevs := elev10-azim30 elev90-azim0
files := $(foreach lat,$(lats),$(foreach azim_elev,$(azim_elevs),$(filename_template)))

animatedLatent/latent0-elev10-azim30_frame1.png: 
	rm -rf animatedLatent/*
	jupyter nbconvert --execute --to notebook VisualizationAnimation.ipynb 

videos: animatedLatent/latent0-elev10-azim30_frame1.png
	rm -rf $(files)
	$(foreach lat, $(lats), $(foreach azim_elev, $(azim_elevs), $(convert_video);))

