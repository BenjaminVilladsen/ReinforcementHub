def print_text_with_border(text: str, px=10, py=1) -> None:
    #half of px and py but as integers
    px_half = int(px / 2)
    py_half = int(py / 2)
    #print the top border
    print(f"+{'-' * (len(text) + px)}+")
    for _ in range(py_half):
        print(f"|{' ' * (len(text) + px)}|")

    #print the text with side borders
    print(f"|{' ' * px_half}{text}{' ' * px_half}|")

    for _ in range(py_half):
        print(f"|{' ' * (len(text) + px)}|")

    #print the bottom border
    print(f"+{'-' * (len(text) + px)}+")


