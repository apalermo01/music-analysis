CREATE database music;
CREATE TABLE train (
	id int not null auto_increment,
    fcc TEXT,
    mfcc TEXT,
    filename varchar(100),
    folder_id int,
    target int,
    PRIMARY KEY(id)
    );