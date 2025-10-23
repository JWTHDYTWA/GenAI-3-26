import argparse
import os
import csv
import GenAI_1_17.main as gai17
import GenAI_2_21.main as gai21
from typing import Literal

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_CSV = "generated.csv"
MAX_OUT_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.2

POST_LEN_LIMIT = 200
SYS_PROMPT_TEMPLATE = 'Твоя задача - написать короткую историю, соблюдая требования:\n' \
                      '* Стиль: {style}\n' \
                      '* Ограничение на длину: 75-150 символов\n'
STORY_PROMPT = 'Напиши историю про {0}, используя эти слова в тексте.'
HASHTAG_PROMPT = 'Выбери и напиши через пробел несколько хэштегов для истории, начиная с самых релевантных:\n\n{0}'


def parse_hashtags(string: str, format: Literal['comma', 'space', 'hash']):
    """
    Извлекает хэштеги из строки в соответствии с заданным форматом.

    Parameters
    ----------
    string : str
        Строка для парсинга.
    format : {'semicolon', 'space', 'hash'}
        Формат строки с хэштегами.
        - 'semicolon' : хэштеги разделены запятой.
        - 'space' : хэштеги разделены пробелами. К каждому слову, не
          начинающемуся с '#', он будет добавлен. Пример: "#котики собачки".
        - 'hash' : хэштеги разделены пробелами, и учитываются только те
          слова, которые уже начинаются с символа '#'. Пример: "#котики #собачки".

    Returns
    -------
    set[str]
        Множество с извлечёнными и отформатированными хэштегами.
        Если входная строка пуста, возвращается пустое множество.

    Raises
    ------
    TypeError
        Если аргумент `string` не является строкой.
    ValueError
        Если указан неподдерживаемый `format`.
    """

    if not isinstance(string, str):
        raise TypeError('Аргумент string должен быть типа str.')
    if not string:
        return set()

    if format == 'comma':
        return {'#' + v.replace(' ', '') for v in string.split(',')}
    elif format == 'space':
        return {'#' + v.strip().removeprefix('#') for v in string.split()}
    elif format == 'hash':
        return {v.strip() for v in string.split() if v.startswith('#')}
    else:
        raise ValueError('Выбран неподдерживаемый формат ввода.')


def add_hashtags(text: str, hashtags: set[str], lim: int):
    """
    Добавляет хэштеги к строке, не превышая лимит длины.

    Исходное множество `hashtags` будет изменено (опустошено)
    в процессе работы функции, так как элементы извлекаются из него
    с помощью метода `pop()`.

    Parameters
    ----------
    text : str
        Исходная строка, к которой нужно добавить хэштеги.
    hashtags : set[str]
        Множество строк, содержащих хэштеги для добавления.
    lim : int
        Лимит длины итоговой строки.

    Returns:
        Строка с добавленными хэштегами.

    Raises
    ------
    TypeError
        Если `text` не является строкой (`str`) или `hashtags`
        не является множеством (`set`).
    """
    if not isinstance(text, str):
        raise TypeError('Arg `text` must be string.')
    if not isinstance(hashtags, set):
        raise TypeError('Arg `hashtags` should be set of strings.')
    
    while hashtags:
        ht = hashtags.pop()
        if len(text) + len(ht) + 1 < lim:
            text = text + ' ' + ht
        else:
            break
    
    return text


def main():
    parser = argparse.ArgumentParser('Генератор контента для соцсетей')
    parser.add_argument('themes', help='Темы для поста. Используйте кавычки.')
    parser.add_argument('style', help='Стиль поста.')
    parser.add_argument(
        '--csv',
        default=DEFAULT_CSV,
        help=f'Путь к csv-файлу, в который будут записываться посты. По умолчанию - "{DEFAULT_CSV}".')
    args = parser.parse_args()

    assert isinstance(args.themes, str)
    assert isinstance(args.style, str)
    gai21.SYSTEM_PROMPT = SYS_PROMPT_TEMPLATE.format(style = args.style)

    try:
        pipe_instance = gai21.text_pipeline_init(MODEL_NAME)
    except Exception as e:
        print(f'Ошибка при инициализации модели:\n{e}')
        return 1
    
    try:
        base_ht = parse_hashtags(args.themes, 'comma')
        print(base_ht)
    except Exception as e:
        print(f'Ошибка при парсинге базовых хэштегов: {e}')
        base_ht = set()
    
    try:
        ### Генерация основной части поста
        # Генератор сообщения из GenAI-2-21 использует системный промпт, а из GenAI-1-17 нет
        messages = gai21.format_message(STORY_PROMPT.format(args.themes))
        story = gai17.generate_text(pipe_instance, messages, MAX_OUT_TOKENS, True, TEMPERATURE, TOP_P, REPETITION_PENALTY)
        assert len(story) < 200
        # print('Story:\n\n' + story + '\n')

        ### Генерация хэштегов при помощи LLM
        messages = gai17.chat_prompt(HASHTAG_PROMPT.format(story))
        htags = gai17.generate_text(pipe_instance, messages, MAX_OUT_TOKENS, True, TEMPERATURE, TOP_P, REPETITION_PENALTY)
        htags_set = base_ht | parse_hashtags(htags, 'hash')
        # print('Hashtags:\n\n' + str(htags_set) + '\n') # Debug

        story = story.replace('\n', ' ')
        story = add_hashtags(story, htags_set, POST_LEN_LIMIT)
    except Exception as e:
        print(f'Ошибка при генерации или подготовке текста:\n{e}')
        return 1

    try:
        with open(args.csv, 'a', encoding='utf-8') as f:
            writer = csv.DictWriter(f, ['Theme', 'Style', 'Post'])
            assert isinstance(story, str)
            writer.writerow({'Theme': args.themes, 'Style': args.style, 'Post': story})
    except Exception as e:
        print(f'Ошибка при открытии файла:\n{e}')


if __name__ == '__main__':
    main()