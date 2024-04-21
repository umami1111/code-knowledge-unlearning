# This file is auto-generated by types_generator.py
# fmt: off
from typing import Literal, TypedDict, Optional


class SolrDocument(TypedDict):
    key: str
    type: Literal['work', 'author', 'subject']
    redirects: Optional[list[str]]
    has_fulltext: Optional[bool]
    title: Optional[str]
    title_suggest: Optional[str]
    title_sort: Optional[str]
    subtitle: Optional[str]
    alternative_title: Optional[list[str]]
    alternative_subtitle: Optional[list[str]]
    edition_count: Optional[int]
    edition_key: Optional[list[str]]
    cover_edition_key: Optional[str]
    by_statement: Optional[list[str]]
    publish_date: Optional[list[str]]
    publish_year: Optional[list[int]]
    first_publish_year: Optional[int]
    first_edition: Optional[str]
    first_publisher: Optional[str]
    language: Optional[list[str]]
    number_of_pages_median: Optional[int]
    lccn: Optional[list[str]]
    ia: Optional[list[str]]
    ia_box_id: Optional[list[str]]
    ia_loaded_id: Optional[list[str]]
    ia_count: Optional[int]
    ia_collection: Optional[list[str]]
    oclc: Optional[list[str]]
    isbn: Optional[list[str]]
    ebook_access: Optional[Literal['no_ebook', 'unclassified', 'printdisabled', 'borrowable', 'public']]
    lcc: Optional[list[str]]
    lcc_sort: Optional[str]
    ddc: Optional[list[str]]
    ddc_sort: Optional[str]
    contributor: Optional[list[str]]
    publish_place: Optional[list[str]]
    publisher: Optional[list[str]]
    publisher_facet: Optional[list[str]]
    first_sentence: Optional[list[str]]
    author_key: Optional[list[str]]
    author_name: Optional[list[str]]
    author_alternative_name: Optional[list[str]]
    author_facet: Optional[list[str]]
    subject: Optional[list[str]]
    subject_facet: Optional[list[str]]
    subject_key: Optional[list[str]]
    place: Optional[list[str]]
    place_facet: Optional[list[str]]
    place_key: Optional[list[str]]
    person: Optional[list[str]]
    person_facet: Optional[list[str]]
    person_key: Optional[list[str]]
    time: Optional[list[str]]
    time_facet: Optional[list[str]]
    time_key: Optional[list[str]]
    ratings_average: Optional[float]
    ratings_sortable: Optional[float]
    ratings_count: Optional[int]
    ratings_count_1: Optional[int]
    ratings_count_2: Optional[int]
    ratings_count_3: Optional[int]
    ratings_count_4: Optional[int]
    ratings_count_5: Optional[int]
    readinglog_count: Optional[int]
    want_to_read_count: Optional[int]
    currently_reading_count: Optional[int]
    already_read_count: Optional[int]
    text: Optional[list[str]]
    seed: Optional[list[str]]
    name: Optional[str]
    name_str: Optional[str]
    alternate_names: Optional[list[str]]
    birth_date: Optional[str]
    death_date: Optional[str]
    date: Optional[str]
    work_count: Optional[int]
    top_work: Optional[str]
    top_subjects: Optional[list[str]]
    subject_type: Optional[str]

# fmt: on