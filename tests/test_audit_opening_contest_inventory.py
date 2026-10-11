import unittest
from scripts.audit_opening_contest_inventory import section_structure


def row(title, boxes):
    return '<tr><td class="race__in-running__title">'+title+'</td>'+''.join('<td><sprite-svg name="rug_'+str(b)+'"></sprite-svg></td>' for b in boxes)+'</tr>'


class TestSectionStructure(unittest.TestCase):
    def test_final_placing_and_pir_cannot_supply_first_section(self):
        self.assertFalse(section_structure(row('Final',[2,1])+row('PIR',[1,2]))['first_section_present'])

    def test_actual_section_order_can_differ_from_later_section(self):
        result=section_structure(row('1st Section',[2,1,5])+row('2nd Section',[1,2,5]))
        self.assertTrue(result['first_section_unique_box_order'])
        self.assertTrue(result['section_order_changes'])

    def test_duplicate_missing_and_nonstarter_boxes_fail(self):
        for boxes in ([1,1],[1,9],[],[1]):
            self.assertFalse(section_structure(row('1st Section',boxes))['first_section_unique_box_order'])

    def test_duplicate_label_is_ambiguous(self):
        result=section_structure(row('1st Section',[1,2])*2)
        self.assertTrue(result['duplicate_section_labels'])
        self.assertFalse(result['first_section_unique_box_order'])

    def test_non_section_rugs_never_enter_sequence(self):
        result=section_structure(row('Finish',[7,8])+row('1st Section',[1,2]))
        self.assertEqual(result['first_section_box_count'],2)
