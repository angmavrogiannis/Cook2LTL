from ai2thor.controller import Controller
import time
import math
import cv2

def euclidean_distance(pointA, pointB):
    return math.dist(pointA, pointB)

def rotate_to_object(object_type):
    obj = obj_in_scene(object_type)
    # 0 is arbitrary
    obj_x = obj["position"]["x"]
    obj_z = obj["position"]["z"]

    agent_position = controller.last_event.metadata["agent"]["position"]
    agent_x = agent_position["x"]
    agent_z = agent_position["z"]

    a = euclidean_distance([agent_x, agent_z], [obj_x, obj_z])
    b = euclidean_distance([agent_x, agent_z], [agent_x - 2, agent_z])
    c = euclidean_distance([obj_x, obj_z], [agent_x - 2, agent_z])

    gamma = math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
    # print((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    # print(f"gamma is {gamma}")
    controller.step(action="RotateRight", degrees=gamma)

def rotate(direction):
    controller.step(action=f"Rotate{direction}")

def pick_up(objectId, force):
    controller.step(action="PickupObject",objectId=objectId, forceAction=force)

def toggle_on(objectId, force):
    controller.step(action="ToggleObjectOn", objectId=objectId, forceAction=force)

def toggle_off(objectId, force):
    controller.step(action="ToggleObjectOff", objectId=objectId, forceAction=force)

def slice(objectId, force):
    controller.step(action="SliceObject", objectId=objectId, forceAction=force)

def put(objectId, force):
    controller.step(action="PutObject", objectId=objectId, forceAction=force)

def open(objectId, force):
    controller.step(action="OpenObject", objectId=objectId, forceAction=force)

def close(objectId, force):
    controller.step(action="CloseObject", objectId=objectId, forceAction=force)

def look(direction):
    controller.step(action=f"Look{direction}")

def obj_in_scene(object_type):
    types_in_scene = sorted([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    assert object_type in types_in_scene, "Object not in scene"
    return next(obj for obj in controller.last_event.metadata["objects"] if obj["objectType"] == object_type)

def closest_position(object_position, reachable_positions):
    out = reachable_positions[0]
    min_distance = float("inf")
    for pos in reachable_positions:
        # NOTE: y is the vertical direction, so only care about the x/z ground positions
        dist = sum([(pos[key] - object_position[key]) ** 2 for key in ["x", "z"]])
        if dist < min_distance:
            min_distance = dist
            out = pos
    return out

def tp_to_object(object_type):
    obj = obj_in_scene(object_type)
    reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    closest = closest_position(obj["position"], reachable_positions)
    controller.step(action="Teleport", **closest)
    rotate_to_object(object_type)
    return obj["objectId"]

if __name__ == "__main__":
    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan1",

        # step sizes
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,

        # image modalities
        renderDepthImage=False,
        renderInstanceSegmentation=False,

        # camera properties
        width=600,
        height=600,
        fieldOfView=90
    )
    step_size = 0.0005
    for i in range(500):
        controller.step(action="MoveAhead", moveMagnitude=i*step_size)
        time.sleep(0.02)

    breakpoint()
    # tp_to_object("Knife")
    # controller.step(action="LookDown")
    # breakpoint()
    # event = controller.step(action="MoveAhead")
    # objects = [object["objectType"] for object in event.metadata['objects']]
    # print(objects)


    # controller.step(action="MoveAhead")
    # breakpoint()
    # obj_id = tp_to_object("Bowl")
    # breakpoint()
    # controller.step(action="PickupObject", objectId=obj_id, forceAction=True)
    # breakpoint()

# controller.step(action='Teleport', position=dict(x=-2.5, y=0.900998235, z=-3.0))
# controller.step(action='LookDown')
# event = controller.step(action='RotateLeft', degrees=180)
# # In FloorPlan28, the agent should now be looking at a mug
# for o in event.metadata['objects']:
#     if o['visible'] and o['pickupable'] and o['objectType'] == 'Mug':
#         event = controller.step(action='PickupObject', objectId=o['objectId'])
#         mug_object_id = o['objectId']
#         break
# breakpoint()
# # the agent now has the Mug in its inventory
# # to put it into the Microwave, we need to open the microwave first
# time.sleep(2)
# event = controller.step(action='LookUp')
# time.sleep(2)
# event = controller.step(action='RotateLeft')
# time.sleep(2)

# event = controller.step(action='MoveLeft')
# event = controller.step(action='MoveLeft')
# event = controller.step(action='MoveLeft')
# event = controller.step(action='MoveLeft')
# time.sleep(2)

# event = controller.step(action='MoveAhead')
# event = controller.step(action='MoveAhead')
# event = controller.step(action='MoveAhead')
# event = controller.step(action='MoveAhead')
# event = controller.step(action='MoveAhead')
# event = controller.step(action='MoveAhead')
# time.sleep(2)
# breakpoint()
# for o in event.metadata['objects']:
#     if o['visible'] and o['openable'] and o['objectType'] == 'Microwave':
#         event = controller.step(action='OpenObject', objectId=o['objectId'])
#         receptacle_object_id = o['objectId']
#         break

# event = controller.step(dict(
#     action='PutObject',
#     receptacleObjectId=receptacle_object_id,
#     objectId=mug_object_id), raise_for_failure=True)

# # close the microwave
# event = controller.step(dict(
#     action='CloseObject',
#     objectId=receptacle_object_id), raise_for_failure=True)