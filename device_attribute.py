class Device:
    def __init__(self, device_id, default_attributes=None):
        self.device_id = device_id
        if default_attributes:
            self.attributes = default_attributes.copy()
        else:
            self.attributes = {'position': [0.0, 0.0, 0.0], 'speed': 0.0}

    def add_attribute(self, attribute_name, attribute_value):
        if attribute_name == 'position':
            if len(attribute_value) != 3:
                print("Error: Position attribute must have three values.")
                return
        elif attribute_name == 'speed':
            if not isinstance(attribute_value, float):
                print("Error: Speed attribute must be a single float value.")
                return
        self.attributes[attribute_name] = attribute_value

    def remove_attribute(self, attribute_name):
        if attribute_name in self.attributes:
            del self.attributes[attribute_name]

    def get_attribute(self, attribute_name):
        return self.attributes.get(attribute_name, None)

    def get_all_attributes(self):
        return self.attributes


class DeviceLibrary:
    def __init__(self):
        self.devices = {}

    def add_device(self, device_id, device):
        self.devices[device_id] = device

    def remove_device(self, device_id):
        if device_id in self.devices:
            del self.devices[device_id]

    def get_device_attribute(self, device_id, attribute_name):
        if device_id in self.devices:
            return self.devices[device_id].get_attribute(attribute_name)
        else:
            return None

    def get_device_all_attributes(self, device_id):
        if device_id in self.devices:
            return self.devices[device_id].get_all_attributes()
        else:
            return {}

if __name__=='__main__':
    # 示例用法
    device_library = DeviceLibrary()

    # 创建设备并添加到属性库中
    device1 = Device("device1")
    device2 = Device("device2")
    device_library.add_device("device1", device1)
    device_library.add_device("device2", device2)

    # 给设备添加属性
    print(device_library.get_device_attribute("device1", "position"))  # 输出: [0, 0, 0]
    print(device_library.get_device_attribute("device2", "position"))  # 输出: [0, 0, 0]

    # 设置新的属性值
    device1.add_attribute("position", [10, 5, 2])
    device1.add_attribute("speed", 4.1)
    device1.add_attribute("position", [10])
    device1.add_attribute("speed", [10, 5])  # 输出错误信息：Error: Speed attribute must be a single float value.
    device2.add_attribute("position", [5, 7, 2])

    # 查询属性
    print(device_library.get_device_attribute("device1", "position"))  # 输出: [10, 5, 2]
    print(device_library.get_device_attribute("device2", "position"))  # 输出: [5, 7, 2]

    # 删除属性
    device1.remove_attribute("position")
    device2.remove_attribute("position")

    # 获取所有属性
    print(device_library.get_device_all_attributes("device1"))  # 输出: {'speed': 4.1}
    print(device_library.get_device_all_attributes("device2"))  # 输出: {'speed': 0.0}
