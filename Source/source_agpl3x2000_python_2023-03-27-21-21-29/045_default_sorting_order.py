from datetime import datetime

from sqlalchemy import MetaData
from sqlalchemy import Column, ForeignKey, Table
from sqlalchemy import Boolean, DateTime, Integer, Unicode, String, UnicodeText

metadata = MetaData()

user_table = Table('user', metadata,
    Column('id', Integer, primary_key=True),
    Column('user_name', Unicode(255), nullable=False, unique=True, index=True),
    Column('display_name', Unicode(255), nullable=True, index=True),
    Column('bio', UnicodeText(), nullable=True),
    Column('email', Unicode(255), nullable=True, unique=False),
    Column('email_priority', Integer, default=3),
    Column('activation_code', Unicode(255), nullable=True, unique=False),
    Column('reset_code', Unicode(255), nullable=True, unique=False),
    Column('password', Unicode(80), nullable=False),
    Column('locale', Unicode(7), nullable=True),
    Column('create_time', DateTime, default=datetime.utcnow),
    Column('access_time', DateTime, default=datetime.utcnow,
           onupdate=datetime.utcnow),
    Column('delete_time', DateTime),
    Column('banned', Boolean, default=False),
    Column('no_help', Boolean, default=False, nullable=True),
    Column('page_size', Integer, default=10, nullable=True)
    )


def upgrade(migrate_engine):
    metadata.bind = migrate_engine

    user_state = Column('proposal_sort_order', Unicode(50), default=None, nullable=True)
    user_state.create(user_table)


def downgrade(migrate_engine):
    raise NotImplementedError()
