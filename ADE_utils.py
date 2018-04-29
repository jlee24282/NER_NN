_author_ = 'Kahyun Lee'
_email_ = 'klee70@gmu.edu'

import os, random
from lxml import etree as ET


def DrugNames(folder):
    DrugList = []
    for fname in os.listdir(folder):
        DrugList.append(fname.rstrip('.xml'))
    return DrugList


def ReportID_extractor(folder):
    ReportID_Drug = {}
    for file in os.listdir(folder):
        fpath = folder + '/' + file
        tree = ET.parse(fpath, parser=ET.XMLParser(remove_blank_text=True))
        reports = tree.xpath('safetyreport')
        for report in reports:
            ID = report.find('safetyreportid').text
            drugs = []
            for drug in report.findall('.//drug/medicinalproduct'):
                brand = drug.text
                if ',' in brand:
                    multibrand = brand.split(',')
                    for item in multibrand:
                        drugs.append(brand.replace('.', ''))
                else:
                    drugs.append(brand.replace('.', ''))
            ReportID_Drug[ID] = drugs
    return ReportID_Drug


def SampleReport(Brand_ReportID):
    RandomReports = {}
    for brand, reportIDs in Brand_ReportID.items():
        ListIDs = list(reportIDs)
        sample = []
        i = 0
        if len(ListIDs) <= 20:
            RandomReports[brand] = ListIDs
        else:
            while i < 20:
                sample.append(random.choice(ListIDs))
                i += 1
            RandomReports[brand] = sample
    return RandomReports


UpdateList = ['ACTEMRA', 'COARTEM', 'COMETRIQ', 'KYPROLIS', 'LATUDA', 'LYNPARZA', 'MOVANTIK', 'OFEV', 'OPDIVO',
              'POMALYST', 'PRISTIQ EXTENDED-RELEASE', 'STELARA', 'ULORIC']

if __name__ == '__main__':
    # DrugList = DrugNames('./data')
    # for item in DrugList:
    # 	if item in UpdateList:
    # 		print(item)
    ReportID_Drug = ReportID_extractor('./FAERS_Reports')
    Brand_ReportID = {}
    for drug in UpdateList:
        ReportSet = set()
        for key, value in ReportID_Drug.items():
            for brand in value:
                if brand == drug:
                    ReportSet.add(key)
                    break
        Brand_ReportID[drug] = ReportSet
    # for key,value in Brand_ReportID.items():
    # 	print(key,value)
    FOIA_request = SampleReport(Brand_ReportID)
    p
    for key, value in FOIA_request.items():
        print(key, value)
