# -*- coding: utf-8 -*-
# (C) 2019 Smile (<https://www.smile.eu>)
# License AGPL-3.0 or later (https://www.gnu.org/licenses/agpl).

from odoo import api, fields, models, _
from odoo.exceptions import ValidationError
import json

# PostgreSQL commands list
_UNSAFE_SQL_KEYWORDS = [
    'ABORT', 'ALTER',
    'BEGIN',
    'CALL', 'CHECKPOINT', 'CLOSE', 'CLUSTER', 'COMMIT', 'COPY', 'CREATE',
    'DEALLOCATE', 'DECLARE', 'DELETE', 'DISCARD', 'DO', 'DROP',
    'END', 'EXECUTE', 'EXPLAIN',
    'FETCH',
    'GRANT',
    'IMPORT', 'INSERT',
    'LISTEN', 'LOAD', 'LOCK',
    'MOVE',
    'PREPARE',
    'REASSIGN', 'REFRESH', 'REINDEX', 'RELEASE', 'RESET', 'REVOKE', 'ROLLBACK',
    'SAVEPOINT', 'SECURITY', 'SET', 'SHOW', 'START',
    'TRUNCATE',
    'UNLISTEN', 'UPDATE',
    'VACUUM', 'VALUES',
]


class Base(models.AbstractModel):
    _inherit = 'base'

    def _valid_field_parameter(self, field, name):
        return name == 'data_mask' or super()._valid_field_parameter(
            field, name)


class IrModelFields(models.Model):
    _inherit = 'ir.model.fields'

    data_mask = fields.Char()
    data_mask_locked = fields.Boolean()

    @api.constrains('data_mask')
    def _check_data_mask(self):

        def _format(string):
            return " %s " % string.lower()

        for field in self:
            if field.data_mask:
                if ';' in field.data_mask:
                    raise ValidationError(
                        _("You cannot use ';' character into a data mask"))
                for unsafe_keyword in _UNSAFE_SQL_KEYWORDS:
                    if _format(unsafe_keyword) in _format(field.data_mask):
                        raise ValidationError(
                            _("You cannot use '%s' keyword into a data mask")
                            % unsafe_keyword)

    def _reflect_field_params(self, field, model_id):
        vals = super(IrModelFields, self)._reflect_field_params(
            field, model_id)
        vals['data_mask'] = getattr(field, 'data_mask', None)
        return vals

    def _instanciate_attrs(self, field_data):
        attrs = super(IrModelFields, self)._instanciate_attrs(field_data)
        if attrs and field_data.get('data_mask'):
            attrs['data_mask'] = field_data['data_mask']
        return attrs

    def toggle_data_mask_locked(self):
        for field in self:
            field.data_mask_locked = not field.data_mask_locked

    _safe_attributes = ['data_mask', 'data_mask_locked']

    def write(self, vals):
        for attribute in self._safe_attributes:
            if attribute in vals:
                fields_to_update = self
                if attribute == 'data_mask':
                    fields_to_update = self.filtered(
                        lambda field: not field.data_mask_locked)
                fields_to_update._write({attribute: vals[attribute]})
                del vals[attribute]
        return super(IrModelFields, self).write(vals)

    @api.model
    def get_anonymization_query(self):
        return self.search([
            ('data_mask', '!=', False),
            ('store', '=', True),
        ])._get_anonymization_query()

    def _get_anonymization_query(self):
        query = "DELETE FROM ir_attachment WHERE name ilike '/web/content/%'" \
                "OR name ilike '%/static/%';\n"
        data = {}
        for field in self:
            if field.data_mask:
                data_mask = "jsonb_build_object('en_US', %s::varchar)" % field.data_mask if field.translate and field.data_mask != "NULL" else field.data_mask
                if self.env[field.model]._table not in data.keys():
                    data[self.env[field.model]._table] = [
                        "UPDATE %s SET %s = %s" % (
                            self.env[field.model]._table,
                            field.name, data_mask
                        )]
                else:
                    if 'where'.lower() in field.data_mask.lower():
                        data[self.env[field.model]._table].append(
                            "UPDATE %s SET %s = %s" % (
                                self.env[field.model]._table,
                                field.name, data_mask
                            ))
                    else:
                        data[self.env[field.model]._table][0] += ",%s = %s" % (
                            field.name, data_mask)


        for val in data.values():
            query += ";\n".join(val) + ";\n"
        return query
