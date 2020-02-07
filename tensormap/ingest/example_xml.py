import tensormap.ingest
import tensormap.utils.ingest
import pathlib
import json
import glob

# import pdb

def example1():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml.zst'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.build(preset = 'ukbb_ecg', compression = 'zstd')
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example2():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.build(preset = 'ukbb_ecg', compression = 'none')
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example3():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml.zst'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.build(preset = 'ukbb_ecg', compression = 'zstd')
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example4():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/muse_ecg.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = False # Import no other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example5():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/muse_ecg.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = True # Import all other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example6():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/ukbb_ecg.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = False # Import all other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example7():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/ukbb_scrubbed.xml'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/ukbb_ecg.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = True # Import all other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example8():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/muse_ecg_data.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = False # Import no other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example9():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/muse_ecg_data.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = True # Import all other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example10():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml.zst'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.zstd_decompress,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/muse_ecg_data.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = False # Import no other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


def example11():
    cur_path = pathlib.Path(__file__).parent.absolute()
    filename = str(cur_path) + '/../data/ingest/ecg/muse_ecg_deidentified_1.xml.zst'
    imp = tensormap.ingest.importer.XmlImporter(filename)
    imp.output_file = '/'.join(["./", imp.automated_name()])
    imp.input_callbacks = [tensormap.utils.ingest.open_read_load,
                           tensormap.utils.ingest.zstd_decompress,
                           tensormap.utils.ingest.xml_parse]

    # JSON mapper file
    with open(str(cur_path) + '/../data/ingest/maps/muse_ecg_data.json') as r:
        mapper = json.loads(r.read())

    imp.mapper = mapper
    imp.mapper_only = True # Import all other fields

    imp.build()
    imp.voltage_postprocessing()
    print(imp._input_file)
    print(imp.retrieve_processed_data())


if __name__ == "__main__":    
    example1()
    example2()
    example3()
    example4()
    example5()
    # example6() # Broken
    example7()
    example8()
    example9()
    example10()
    example11()

