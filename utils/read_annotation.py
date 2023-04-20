import json
from pathlib import Path

annotationsDir = Path('../annotations')
aggregatedAnnotations = dict()

try:
    for x in annotationsDir.iterdir():
        if x.suffix.lower() == ".json":
            with open(x, 'r') as f:
                annotationDict = json.load(f)
                annList = list(map(dict, annotationDict.values()))
                image_name_to_class_map = { it['filename'] : it['regions'][0]['region_attributes']['class'].lower() for it in annList }
                aggregatedAnnotations.update(image_name_to_class_map)

    image_name_to_class_file = '../annotations/aggragatedAnnotations.txt'
    with open(image_name_to_class_file, 'w') as json_file:
        json.dump(aggregatedAnnotations, json_file, indent = 4)
except:
    raise Exception("Failed to aggregate annotated json files")
finally:
    print("Annotation files aggragated successfully! Results were written to {}".format(image_name_to_class_file))
    