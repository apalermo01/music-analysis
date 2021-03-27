# How to run these scripts
First the sql_scripts can be run within Mysql
* 1-3 add a music database and test, train, validate tables
Add the actual data to the music tables using the following 3 files which must be in the same folder as the .JSON file included with the nsynth dataset.
* add_*table name*.py
* get_*table name*.py
* fft_mfcc.py
The last script can be run at this point. I am trying to make it so these steps can be skipped by making the tables locally and uploading a dump of each table. So far, I have done that up to this point. The next step is to add an alternate_target column to each table and a parent table for those targets which is based on the first 12 chars of the original wav filename.
* 5_Update_alternate_target.sql
