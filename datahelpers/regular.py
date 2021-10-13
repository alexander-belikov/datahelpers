import re
from functools import reduce

list_to_strip_sing = ["protein", "gene", "cell", "enzyme"]
list_to_strip = list_to_strip_sing + list(map(lambda x: x + "s", list_to_strip_sing))


def eat_words_from_list(z, list_of_words=list_to_strip):

    """

    this regexp matches any number of 'y ' in the beginning of the string
    and any number of ' y' at the end of the string

    :param z: string
    :param list_of_words: list of strings
    :return: resulting string

    """
    return reduce(
        lambda x, y: re.sub("(\.*( " + y + ")*$|^(" + y + " )*\.*)", "", x),
        list_of_words,
        z,
    )


def eat_spaces(s):
    """
    cut extra spaces, more than one in the middle, all on the left and on the right

    :param s:
    :return:
    """
    return re.sub(" +", " ", s).rstrip().lstrip()


def eat_dash(s):
    """
    cut out a dash between alpha and numeric
        example: 'oct-434' -> 'oct434'
    """
    return re.sub(r"(?<=\w)+(-)(?=[0-9])+", "", s)


eating_funcs = [eat_words_from_list, eat_spaces, eat_dash]


def chain_regexp_transforms(z):
    """

    :param z:
    :return:
    """
    return reduce(lambda x, f: f(x), eating_funcs, z)
