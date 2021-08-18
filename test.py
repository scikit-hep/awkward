

def remove_duplicates():
    print("Removing Duplicates from Output File")
    file = open("testing2.txt", "r")
    lines = file.readlines()
    unique = set()
    for line in lines:
        if line[0:5] == "carry":
            unique.add(line)
    file.close()
    file = open("testing2.txt", "w").writelines(unique)

remove_duplicates()