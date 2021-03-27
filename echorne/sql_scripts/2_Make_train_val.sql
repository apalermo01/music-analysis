CREATE TABLE train (
	id int not null auto_increment,
    fft TEXT,
    mfcc TEXT,
    filename varchar(100),
    folder_id int,
    target int,
    PRIMARY KEY(id)
    );
CREATE TABLE validate (
	id int not null auto_increment,
    fft TEXT,
    mfcc TEXT,
    filename varchar(100),
    folder_id int,
    target int,
    PRIMARY KEY(id)
    );