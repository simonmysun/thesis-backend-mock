import random
import paho.mqtt.client as mqtt

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("tele/indoor_sound_classification/+/state")
    topic = "paho/t"
    message = "b"
    print("sending \"%s\" on \"%s\"" % (topic, message))
    client.publish(topic, message)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print("received on \"%s\" : \"%s\" " % (msg.topic, str(msg.payload)))

client_id = 'data_source_py_%030x' % random.randrange(16 ** 30)
print("Initializing MQTT Client %s" % client_id)
client = mqtt.Client(client_id)
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set(username="TuCDataSource", password="b52393dda08af7a991e7af074f377997a082f652")

client.connect("mqtt.makelove.expert", port=1883, keepalive=60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
