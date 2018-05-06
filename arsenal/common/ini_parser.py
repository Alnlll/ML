#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
from configobj import ConfigObjError
from configobj import ConfigObj as Parser

InputError = -1

class IniParser(object):

    def __init__(self, config_file):

        if os.path.exists(config_file):
            try:
                self.config = Parser(config_file, raise_errors=True)
                self.filename = self.config.filename
            except ConfigObjError as e:
                print('IniParser -- {}'.format(e))
                return None
        else:
            print('Cannot find config file: %s', config_file)
            return None
    def translateToDict(self):
        return self.config

    def getValue(self, section, option):
        try:
            self.config[section][option]
        except KeyError:
            print('Cannot get [%s]: %s.' % (section, option))
            return None

        return self.config[section][option]

    def setValue(self, section, option, value):
        try:
            self.config[section][option]
        except KeyError:
            print('Cannot find [%s]: %s.' % (section, option))
            return False

        self.config[section][option] = value
        self.config.write()

        return self.config[section][option]

    def addOption(self, section, option, value):
        try:
            self.config[section]
        except KeyError:
            print('Cannot find [%s].' % section)
            return None
        try:
            self.config[section][option]
        except KeyError:
            self.config[section][option] = value
            self.config.write()
            return self.config[section][option]

        print('[%s]: %s already in file and we will modify it for you.' % (section, option))
        return None

    def delOption(self, section, option):

        try:
            del self.config[section][option]
        except KeyError:
            print('Cannot find [%s]: %s in config file.' % (section, option))
            return False
        self.config.write()

        return True

    def getSection(self, section):
        try:
            self.config[section]
        except KeyError:
            print('Cannot find [%s] in config file.' % section)
            return None

        return self.config[section]

    def delSection(self, section):
        try:
            del self.config[section]
        except KeyError:
            print('Cannot find [%s] in config file.' % section)
            return False

        self.config.write()

        return True
    def exists(self, section = None, option = None):
        if option and section:
            try:
                self.config[section][option]
            except KeyError:
                return False
            return True

        elif section and (not option):
            try:
                self.config[section]
            except KeyError:
                return False
            return True
        else:
            print('section:%r option:%r illegal.' % (section, option))
            return InputError


    def renameSection(self, old_section, new_section):
        try:
            self.config[old_section]
        except KeyError:
            print('Cannot find section:%s', old_section)
            return False

        self.config.rename(old_section, new_section)
        self.config.write()

        return True

    def copyToAnotherFile(self, new_file):
        if os.path.exists(new_file):
            print('%s already exists.' % new_file)

        self.config.filename = new_file
        self.config.write()
        self.config.filename = self.filename

    def mergeIniFile(self, merge_obj):
        '''merge another ini file into current file'''
        if merge_obj:
            self.config.merge(merge_obj.dict())
        else:
            return False

        self.config.write()

        return True

if __name__ == "__main__":
    pass
'''
    config = IniParser('config.ini')
    print config.getValue('CONFIG', 'instruIP')
    if config.getValue('CONFIG', 'instruIP') == '192.168.100.230':
        print 'getValue test ok!'
    if config.setValue('CONFIG', 'instruIP', '192.168.100.122') == '192.168.100.122':
        print 'setValue test ok!'
    if config.addOption('CONFIG', 'test', 'test') == 'test':
        print 'add existed section Ok!'
    if not config.addOption('TEST', 'test', 'test') == 'test':
        print 'add unexisted section Ok!'
    if config.delOption('CONFIG', 'gainDef'):
        print 'del option OK'
    config.delSection('PATHLOSS')
    config.renameSection('CONFIG', 'CONFIG_TEST')
    config.copyToAnotherFile('config_test.ini')
    print config.config
    config1 = IniParser('config1.ini')
    print config1.config
    config.mergeIniFile(config1.config)
'''
