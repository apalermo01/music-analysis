CREATE database music;
CREATE TABLE test (
	id int not null auto_increment,
    fft TEXT,
    mfcc TEXT,
    filename varchar(100),
    folder_id int,
    target int,
    PRIMARY KEY(id)
    );
