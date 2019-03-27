classes = {}


def prompt():
    print(classes)
    print('MENU')
    print('1 --> Add classes.')
    print('2 --> Remove classes.')
    print('3 --> View course details.')
    print('4 --> View schedule.')
    print('5 --> Quit.')
    option = int(input('What would you like to choose? '))
    while option < 1 or option > 5:
        option = int(input('Please enter a valid numerical option: '))
    return option


def checkOption(option):
    global classes
    if option == 1:
        courseCount = courseCounter()
        classes = addClasses(courseCount)
    elif option == 2:
        removeClasses()
    elif option == 3:
        viewCourseDetails()
    elif option == 4:
        viewSchedule()


def courseCounter():
    courseCount = input('Enter a numerical value of courses only (up to 4): ')
    while not courseCount.isnumeric():
        courseCount = input('Enter a NUMERICAL value of courses only (up to 4): ')
    return int(courseCount)


def addClasses(courseCount):
    i = 1
    while i <= courseCount:
        courseName = input('Enter a course name: ')
        classes[courseName] = {}
        classes[courseName]['Room Number'] = input('Enter a room number: ')
        classes[courseName]['Instructor'] = input('Enter a instructor: ')
        classes[courseName]['Meeting time'] = input('Enter a meeting time: ')
        i = i + 1
    return classes


def removeClasses():
    courseName = input('Enter the class you would like to remove: ')
    del classes[courseName]
    print(classes)


def main():
    option = prompt()
    checkOption(option)
    while option > 1 or option < 5:
        if option == 5:
            break
        option = prompt()
        checkOption(option)


main()
