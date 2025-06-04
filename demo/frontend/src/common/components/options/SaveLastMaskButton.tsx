import React, { useState, useCallback } from 'react';
import { useAtomValue } from 'jotai';
import { Save } from '@carbon/icons-react';
import OptionButton from './OptionButton'; // Assumes it's in the same 'options' directory
import { sessionAtom } from '@/demo/atoms';
import useMessagesSnackbar from '@/common/components/snackbar/useDemoMessagesSnackbar'; // Using the demo-specific snackbar

export default function SaveLastMaskButton() {
  const session = useAtomValue(sessionAtom);
  const { enqueueMessage } = useMessagesSnackbar();
  const [isLoading, setIsLoading] = useState(false);

  const handleClick = useCallback(async () => {
    if (!session?.id) {
      return;
    }

    setIsLoading(true);
    try {
      // Consider prepending with an API base URL if your setup requires it
      // e.g., `${settings.inferenceAPIEndpoint}/session/${session.id}/save_last_mask`
      const response = await fetch(`/session/${session.id}/save_last_mask`, {
        method: 'POST',
      });

    } catch (error) {
      console.error('Error saving mask:', error);
    } finally {
      setIsLoading(false);
    }
  }, [session, enqueueMessage]);

  const isDisabled = !session?.id || isLoading;

  return (
    <OptionButton
      title="Save Last Mask"
      Icon={Save}
      onClick={handleClick}
      isDisabled={isDisabled}
      loadingProps={{ loading: isLoading, label: 'Saving...' }}
    // variant="default" // Default variant is fine
    />
  );
}
