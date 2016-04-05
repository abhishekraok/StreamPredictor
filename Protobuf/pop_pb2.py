# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pop.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='pop.proto',
  package='streampredictor',
  serialized_pb=_b('\n\tpop.proto\x12\x0fstreampredictor\"\xaf\x01\n\x03Pop\x12\x18\n\x10unrolled_pattern\x18\x01 \x01(\t\x12\x10\n\x08strength\x18\x02 \x01(\x03\x12\x17\n\x0f\x66irst_component\x18\x03 \x01(\t\x12\x18\n\x10second_component\x18\x04 \x01(\t\x12\x1b\n\x13\x66irst_child_parents\x18\x05 \x03(\t\x12\x12\n\nbelongs_to\x18\x06 \x01(\t\x12\x18\n\x10\x63\x61tegory_members\x18\x07 \x03(\t\">\n\nPopManager\x12\x30\n\x12pattern_collection\x18\x01 \x03(\x0b\x32\x14.streampredictor.Pop')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_POP = _descriptor.Descriptor(
  name='Pop',
  full_name='streampredictor.Pop',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='unrolled_pattern', full_name='streampredictor.Pop.unrolled_pattern', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='strength', full_name='streampredictor.Pop.strength', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_component', full_name='streampredictor.Pop.first_component', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='second_component', full_name='streampredictor.Pop.second_component', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='first_child_parents', full_name='streampredictor.Pop.first_child_parents', index=4,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='belongs_to', full_name='streampredictor.Pop.belongs_to', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='category_members', full_name='streampredictor.Pop.category_members', index=6,
      number=7, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=206,
)


_POPMANAGER = _descriptor.Descriptor(
  name='PopManager',
  full_name='streampredictor.PopManager',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pattern_collection', full_name='streampredictor.PopManager.pattern_collection', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=208,
  serialized_end=270,
)

_POPMANAGER.fields_by_name['pattern_collection'].message_type = _POP
DESCRIPTOR.message_types_by_name['Pop'] = _POP
DESCRIPTOR.message_types_by_name['PopManager'] = _POPMANAGER

Pop = _reflection.GeneratedProtocolMessageType('Pop', (_message.Message,), dict(
  DESCRIPTOR = _POP,
  __module__ = 'pop_pb2'
  # @@protoc_insertion_point(class_scope:streampredictor.Pop)
  ))
_sym_db.RegisterMessage(Pop)

PopManager = _reflection.GeneratedProtocolMessageType('PopManager', (_message.Message,), dict(
  DESCRIPTOR = _POPMANAGER,
  __module__ = 'pop_pb2'
  # @@protoc_insertion_point(class_scope:streampredictor.PopManager)
  ))
_sym_db.RegisterMessage(PopManager)


# @@protoc_insertion_point(module_scope)
