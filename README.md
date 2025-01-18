# Final Capstone Project

## Project Description

This final capstone project assignment was completed in Spring 2024. The goal was to replicate tasks that may be asked as a data scientists working at Spotify and gain a deeper understanding of what makes music popular, as well as the audio features associated with specific genres.

The project consists of answering several open-ended questions by analyzing a dataset consisting of 52,000 songs from Spotify. The questions asked can be found following the link given below. Each question required:

1. A detailed explanation of the insights or conclusions derived
2. A clear methodology
3. A figure or statistic to illustrate the findings

Topics covered for this project include data preprocessing techniques, correlation measures, parametric and nonparametric significance tests, linear regression, logistic regression, and PCA dimensional reduction.

## Dataset

The dataset provided consists of 52,000 randomly selected songs on Spotify. Each song has 20 feature columns provided by the Spotify API, detailed below:

1. songNumber - track ID (0 - 51999)
2. artists - artist credited for the song
3. album_name - name of the album
4. track_name - song title
5. popularity - integer metric of how much the song was played (0-100)
6. duration - song duration in milliseconds
7. explicit - binary variable whether the lyrics contain explicit language (1) or not (0)
8. danceability - numerical metric for how easy it is to dance to the song (0-1)
9. energy - numerical metric of song intensity (0-1)
10. key - categorical value of what key the song is in (A-G# : 0-11)
11. loudness - average volume of song in decibels
12. mode - binary variable whether the song is in major key (1) or not (0)
13. speechiness - numerical metric of how much of the song is spoken (0-1)
14. acousticness - numerical metric of how much of the song includes synthesized sounds or acoustic instruments (0-1)
15. instrumentalness - numerical metric of how much of the song is solely intrumental (0-1)
16. liveness - numerical metric for how likely the song was recorded with live audience or in studio (0-1)
17. valence - numerical metric for the positive or negative mood of the song (0-1)
18. tempo - song beats per minute (BPM)
19. time_signature - beats per measure
20. track_genre - genre assigned by Spotify

The csv file used is provided: [spotify52kData.csv](data/spotify52kData.csv).

## File Roadmap

Project Instructions/Questions: [CapstoneProjectSpecSheet.pdf](CapstoneProjectSpecSheet.pdf)

Detailed Response and Aubmission: [project/capstone_response.pdf](project/capstone_response.pdf)

Code Used to Answer Questions (Python): [project/capstone.py](project/capstone.py)

