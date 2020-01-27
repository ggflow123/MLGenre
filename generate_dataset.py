# Yuanzhe Liu
# generate_dataset.py
# Tidy up the Million Song Dataset (Subset) for Machine Learning on Classifying music genre
# Starting from 1/14/2020
# GNU General Public License

import os
import sys
import hdf5_getters as GETTERS
import glob
import h5py
import tables
import numpy as np
import pandas as pd

# iterate through all files
# input: basedir: the directory that the function will work on
# func: the function that will be applied to the files, by default, don't do any changes
# return the number of all the files (counting)
def walk_through_all_files(basedir, attributes, data):
    count = 0 # number of files
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + '.h5'))
        for file in files:
            dic = dict()
            instance = extract_data_into_one_instance(file)
            if len(instance) == 0:
                # print('Empty')
                continue
            else:
                for i in range(len(attributes)):
                    dic[attributes[i]] = instance[i]
                data = data.append(dic, ignore_index=True)
                count += len(files)
                # print(count)
    return count, data

# extract information from each single file in the dataset,
def extract_data_into_one_instance(file):
    h5 = tables.open_file(file, 'r')
    artist_term = np.array(GETTERS.get_artist_terms(h5))
    # print('artist_term')
    # print(artist_term)
    if len(artist_term) == 0:
        # print('Empty in Function')
        h5.close()
        return []

    tempo = GETTERS.get_tempo(h5)
    time_signature = GETTERS.get_time_signature(h5)
    key = GETTERS.get_key(h5)
    loudness = GETTERS.get_loudness(h5)
    mbtags = np.array(GETTERS.get_artist_mbtags(h5))
    # print('mbtags', mbtags, 'tempo', tempo)
    artist_terms = artist_term[0].decode('utf-8')
    '''
    bars_start = np.array(GETTERS.get_bars_start(h5))
    beats_start = np.array(GETTERS.get_beats_start(h5))
    sections_start = np.array(GETTERS.get_sections_start(h5))
    segments_loudness_max = np.array(GETTERS.get_segments_loudness_max(h5))
    segments_loudness_max_time = np.array(GETTERS.get_segments_loudness_max_time(h5))
    segments_pitches = np.array(GETTERS.get_segments_pitches(h5))
    segments_start = np.array(GETTERS.get_segments_start(h5))
    segments_timbre = np.array(GETTERS.get_segments_timbre(h5))
    tatums_start = np.array(GETTERS.get_tatums_start(h5))
    '''

    '''
    instance = [artist_terms, bars_start, beats_start, sections_start,
                segments_loudness_max, segments_loudness_max_time,
                segments_pitches, segments_start, segments_timbre,
                tatums_start, tempo, time_signature]'''

    instance_simple = [artist_terms, tempo, time_signature, key, loudness]
    h5.close()
    # print(attributes)
    # print(instance)

    return instance_simple

def main():
    msd = sys.argv[1] #should be the Million Song Dataset
    msd_data = os.path.join(msd, 'data')
    path1 = os.path.join(msd_data, 'A', 'A')
    count = 0
    current = os.getcwd()
    # os.chdir(msd_data)

    attributes = ['label', 'bars_start', 'beats_start', 'sections_start',
                  'segments_loudness_max', 'segments_loudness_max_time',
                  'segments_pitches', 'segments_start', 'segments_timbre',
                  'tatums_start', 'tempo', 'time_signature']

    attributes_simple = ['label', 'tempo', 'time_signature', 'key', 'loudness']

    data = pd.DataFrame(columns=attributes_simple)

    '''
    for root, dirs, files in os.walk(msd_data):
        files = glob.glob(os.path.join(root, '*'+'.h5'))
        count += len(files)
        for file in files:
            h5 = tables.open_file(file, 'r')
            artist_name = GETTERS.get_artist_name(h5)
            artist_terms = np.array(GETTERS.get_artist_terms(h5))
            artist_terms_fre = np.array(GETTERS.get_artist_terms_freq(h5))
            artist_terms_w = np.array(GETTERS.get_artist_terms_weight(h5))
            n = GETTERS.get_num_songs(h5)
            artist_mbtags = np.array(GETTERS.get_artist_mbtags(h5))


            print(h5.keys())
            analysis = h5['analysis']
            # print(analysis)
            print(h5['analysis'], h5['metadata'], h5['musicbrainz'])
            h5.close()
            print(h5)
            print(artist_terms)
            print(artist_terms_fre)
            print(artist_terms_w)
            s = artist_terms[0].decode('utf-8')
            print(s)
            print(artist_mbtags)
            print(GETTERS.get_segments_loudness_max(h5))
            h5.close()
            return
    '''
    count, newdata = walk_through_all_files(msd_data, attributes_simple, data)

    newdata.to_csv('simple_data.csv', index=None)

    print(count)

if __name__ == '__main__':
    main()