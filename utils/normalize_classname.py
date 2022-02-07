def normalize_classname(class_name):  # normalize the class name to one used by the model
    class_name = class_name.lower()  # remove capitalization
    class_name = class_name.strip()  # remove leading and trailing whitespace
    classes_dict = {  # dictionary containing all class names used in labels and their appropriate model class name
        'gas entrapment porosity': 'gas entrapment porosity',
        'keyhole porosity': 'keyhole porosity',
        'lack of fusion porosity': 'lack of fusion porosity',
        'fusion porosity': 'lack of fusion porosity',
        'gas porosity': 'gas entrapment porosity',
        'lack-of-fusion': 'lack of fusion porosity',
        'keyhole': 'keyhole porosity',
        'other': 'other',
        'lack of fusion': 'lack of fusion porosity',
        'lack-of_fusion': 'lack of fusion porosity'
    }
    return classes_dict.get(class_name)