import json
import os

image = 'shield'

input_schema_path = "./input.schema"
output_schema_path = "./output.schema"
input_path = "./input/input.json"
output_path = "./output/output.json"

assert(os.path.exists(input_schema_path))
assert(os.path.exists(output_schema_path))

input_schema = json.load(open(input_schema_path))
input_json = json.load(open(input_path))
output_schema = json.load(open(output_schema_path))

# docker image build
# os.system("docker build -t %s ." % image)

# check inputs schema
# assert(len(input_schema) == 3)
# for i in range(3):
#     assert("action" in input_schema[i] and "function" in input_schema[i])
#     assert(input_schema[i]["action"] in ["transformation", "train", "evaluation"])
#     l = len(input_schema[i]["function"])
#     for j in range(l):
#         assert("name" in input_schema[i]["function"][j])
#         assert("option" in input_schema[i]["function"][j])
#
#         m = len(input_schema[i]["function"][j]["option"])
#         for k in range(m):
#             assert("name" in input_schema[i]["function"][j]["option"][k])
#             assert("type" in input_schema[i]["function"][j]["option"][k])
#             assert("required" in input_schema[i]["function"][j]["option"][k])
#
# # check input
# assert("action" in input_json)
# assert("name" in input_json)
# assert("num_files" in input_json)
# assert("files" in input_json)
# assert("option" in input_json)
# assert("tags" in input_json)
# assert("model" in input_json)


# docker run
os.system('''docker run \
                    --user 1001 \
                    --mount type=bind,source="%s",target=/app \
                    --mount type=bind,source="%s",target=/mnt/input \
                    --mount type=bind,source="%s",target=/mnt/output \
                    %s:latest''' % (os.path.abspath("."),
                                    os.path.abspath("./input"),
                                    os.path.abspath("./output"),
                                    image,))
