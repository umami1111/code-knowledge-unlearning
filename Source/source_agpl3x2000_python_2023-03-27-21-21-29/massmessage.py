import logging
import textwrap

from pylons import request
from pylons import tmpl_context as c
from pylons.decorators import validate
from pylons.controllers.util import redirect
from pylons.i18n import _

import formencode
from formencode import validators, htmlfill

from adhocracy import config
from adhocracy import forms
from adhocracy.controllers.instance import InstanceController
from adhocracy.lib.auth import require
from adhocracy.lib.auth.authorization import has
from adhocracy.lib.auth.csrf import RequireInternalRequest
from adhocracy.lib import helpers as h
from adhocracy.lib.message import render_body
from adhocracy.lib.message import send as send_message
from adhocracy.lib.base import BaseController
from adhocracy.lib.templating import render, ret_abort, ret_success
from adhocracy.lib.util import get_entity_or_abort
from adhocracy.lib import democracy
from adhocracy.model import Instance
from adhocracy.model import Membership
from adhocracy.model import Permission
from adhocracy.model import User
from adhocracy.model import UserBadge
from adhocracy.model import UserBadges
from adhocracy.model import Proposal

log = logging.getLogger(__name__)


class MassmessageBaseForm(formencode.Schema):
    allow_extra_fields = True
    subject = validators.String(max=140, not_empty=True)
    body = validators.String(min=2, not_empty=True)


class MassmessageForm(MassmessageBaseForm):
    filter_instances = forms.MessageableInstances(not_empty=True)
    filter_badges = forms.ValidUserBadges()
    sender_email = validators.String(not_empty=True)
    sender_name = validators.String(not_empty=False, if_missing=None)
    include_footer = formencode.validators.StringBoolean(if_missing=False)


class MassmessageProposalForm(MassmessageBaseForm):
    creators = validators.StringBool(not_empty=False, if_empty=False,
                                     if_missing=False)
    supporters = validators.StringBool(not_empty=False, if_empty=False,
                                       if_missing=False)
    opponents = validators.StringBool(not_empty=False, if_empty=False,
                                      if_missing=False)
    chained_validators = [
        forms.ProposalMessageNoRecipientGroup(),
    ]


def _get_options(func):
    """ Decorator that calls the functions with the following parameters:
        sender_email - Email address of the sender
        sender_name  - Name of the sender
        subject      - Subject of the message
        body         - Body of the message
        recipients   - A list of users the email is going to
    """
    @RequireInternalRequest(methods=['POST'])
    @validate(schema=MassmessageForm(), form='new')
    def wrapper(self):
        allowed_sender_options = self._get_allowed_sender_options(c.user)
        sender_email = self.form_result.get('sender_email')
        if ((sender_email not in allowed_sender_options) or
                (not allowed_sender_options[sender_email]['enabled'])):
            return ret_abort(_("Sorry, but you're not allowed to set these "
                               "message options"), code=403)
        sender_name = None
        if has('global.message'):
            sender_name = self.form_result.get('sender_name')
        if not sender_name:
            sender_name = c.user.name

        recipients = User.all_q()
        filter_instances = self.form_result.get('filter_instances')
        recipients = recipients.join(Membership).filter(
            Membership.instance_id.in_(filter_instances))
        filter_badges = self.form_result.get('filter_badges')
        if filter_badges:
            recipients = recipients.join(UserBadges,
                                         UserBadges.user_id == User.id)
            recipients = recipients.filter(
                UserBadges.badge_id.in_([fb.id for fb in filter_badges]))

        if has('global.admin'):
            include_footer = self.form_result.get('include_footer')
        else:
            include_footer = True

        if len(filter_instances) == 1:
            instance = Instance.find(filter_instances[0])
        else:
            instance = None

        return func(self,
                    self.form_result.get('subject'),
                    self.form_result.get('body'),
                    recipients.all(),
                    sender_email=allowed_sender_options[sender_email]['email'],
                    sender_name=sender_name,
                    instance=instance,
                    include_footer=include_footer,
                    )
    return wrapper


class MassmessageController(BaseController):
    """
    This deals with messages to multiple users at the same time. This will be
    will be merged with the MessageController at some point.
    """

    @classmethod
    def _get_allowed_instances(cls, user):
        """
        returns all instances in which the given user has permission to send a
        message to all users
        """
        if has('global.message'):
            return Instance.all(include_hidden=True)
        else:
            perm = Permission.find('instance.message')
            instances = [m.instance for m in user.memberships
                         if (m.instance is not None
                             and m.instance.is_authenticated
                             and perm in m.group.permissions)]
            return sorted(instances, key=lambda i: i.label)

    @classmethod
    def _get_allowed_sender_options(cls, user):
        sender_options = {
            'user': {
                'email': user.email,
                'checked': False,
                'enabled': user.is_email_activated(),
                'reason': _("Email isn't activated"),
            },
            'system': {
                'email': config.get('adhocracy.email.from'),
                'checked': False,
                'enabled': config.get_bool(
                    'allow_system_email_in_mass_messages'),
                'reason': _("Not permitted in system settings"),
            },
            'support': {
                'email': config.get('adhocracy.registration_support_email'),
                'checked': False,
                'enabled': (config.get('adhocracy.registration_support_email')
                            is not None),
                'reason': _("adhocracy.registration_support_email not set"),
            }
        }

        if sender_options['user']['enabled']:
            sender_options['user']['checked'] = True
        elif sender_options['system']['enabled']:
            sender_options['system']['checked'] = True

        return sender_options

    def new(self, id=None, errors={}, format=u'html'):

        if id is None:
            require.perm('global.message')
            template = '/massmessage/new.html'
            c.preview_url = h.base_url('/message/preview')
        else:
            c.page_instance = InstanceController._get_current_instance(id)
            require.instance.message(c.page_instance)
            template = '/instance/message.html'
            c.preview_url = h.base_url(
                '/instance/%s/message/preview' % id)

        defaults = dict(request.params)
        defaults.setdefault('include_footer', 'on')

        data = {
            'instances': self._get_allowed_instances(c.user),
            'sender_options': self._get_allowed_sender_options(c.user),
            'userbadges': UserBadge.all(instance=c.instance,
                                        include_global=True)
        }

        return htmlfill.render(render(template, data,
                                      overlay=format == u'overlay'),
                               defaults=defaults, errors=errors,
                               force_defaults=False)

    @_get_options
    def preview(self, subject, body, recipients, sender_email, sender_name,
                instance, include_footer):
        recipients_list = sorted(list(recipients), key=lambda r: r.name)
        if recipients_list:
            try:
                rendered_body = render_body(body, recipients_list[0],
                                            is_preview=True)
            except (KeyError, ValueError) as e:
                rendered_body = _('Could not render message: %s') % str(e)
        else:
            rendered_body = body

        # wrap body, but leave long words (e.g. links) intact
        rendered_body = u'\n'.join(textwrap.fill(line, break_long_words=False)
                                   for line in rendered_body.split(u'\n'))

        data = {
            'sender_email': sender_email,
            'sender_name': sender_name,
            'subject': subject,
            'body': rendered_body,
            'recipients': recipients_list,
            'recipients_count': len(recipients_list),
            'params': request.params,
            'include_footer': include_footer,
            'instance': instance,
        }
        return render('/massmessage/preview.html', data)

    @_get_options
    def create(self, subject, body, recipients, sender_email, sender_name,
               instance, include_footer):
        send_message(subject, body, c.user, recipients,
                     sender_email=sender_email,
                     sender_name=sender_name,
                     instance=instance,
                     include_footer=include_footer)
        return ret_success(
            message=_("Message sent to %d users.") % len(recipients))

    def new_proposal(self, proposal_id, errors={}, format=u'html'):
        c.proposal = get_entity_or_abort(Proposal, proposal_id)
        require.proposal.message(c.proposal)
        defaults = dict(request.params)
        return htmlfill.render(render('/massmessage/new_proposal.html',
                                      overlay=format == u'overlay'),
                               defaults=defaults, errors=errors,
                               force_defaults=False)

    @validate(schema=MassmessageProposalForm(), form='new_proposal')
    def create_proposal(self, proposal_id):
        c.proposal = get_entity_or_abort(Proposal, proposal_id)
        require.proposal.message(c.proposal)

        recipients = set()
        if self.form_result.get(u'supporters'):
            recipients.update(democracy.supporters(c.proposal.rate_poll))
        if self.form_result.get(u'opponents'):
            recipients.update(democracy.opponents(c.proposal.rate_poll))
        if self.form_result.get(u'creators'):
            recipients.update(c.proposal.get_creators())

        send_message(self.form_result.get('subject'),
                     self.form_result.get('body'),
                     c.user,
                     recipients,
                     instance=c.instance)
        h.flash(_("Message sent to %d users.") % len(recipients), 'success')
        redirect(h.entity_url(c.proposal))
