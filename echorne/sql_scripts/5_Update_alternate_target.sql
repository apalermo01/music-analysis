# This can be run after the music_test and music_validate sql files are imported.
CREATE TABLE group_to_id (
id INT4 AUTO_INCREMENT,
instrument VARCHAR(256),
PRIMARY KEY (id));

INSERT INTO group_to_id (instrument)
SELECT DISTINCT left(filename,12) FROM test;

UPDATE test t
JOIN group_to_id g ON left(t.filename,12) = g.instrument
SET t.alternate_target = g.id;

UPDATE validate v
JOIN group_to_id g ON left(v.filename,12) = g.instrument
SET v.alternate_target = g.id;